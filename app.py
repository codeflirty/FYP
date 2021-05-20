# import libraries

from flask import Flask, request
import pyrebase
import speech_recognition as sr
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import path

our_prediction = ''

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        ا 2
        ب 3
        پ 4
        ت 5
        ٹ 6
        ث 7
        ج 8
        چ 9
        ح 10
        خ 11
        د 12
        ڈ 13
        ذ 14
        ر 15
        ڑ 16
        ز 17
        ژ 18
        س 19
        ش 20
        ص 21
        ض 22
        ط 23
        ظ 24
        ع 25
        غ 26
        ف 27
        ق 28
        ک 29
        گ 30
        ل 31
        م 32
        ن 33
        ں 34
        و 35
        ہ 36
        ھ 37
        ی 38
        ے 39
        ئ 40
        ء 41
        آ 42
        اً 43
        ؤ 44
         ّ 45
         ٔ 46
         ً 47
         ُ 48
         َ 49
         ِ 50
         \u200c 51
        """
        self.char_map = {}
        self.index_map = {}
        
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


# In[4]:


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


# In[5]:


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


# In[6]:


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


# In[7]:


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


# In[8]:


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


# In[9]:


# Specify a path
PATH = "Trained_Model.pt"

# Load
trained_model = torch.load(PATH,map_location ='cpu')
trained_model.eval()


# In[10]:


import soundfile as sf

filename = 'download.wav'
wav, _ = sf.read(filename, dtype='float32')


# In[11]:


def GreedyDecoder(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes


# In[12]:


data = torch.from_numpy(wav)
spectrograms = []
spec = train_audio_transforms(data).squeeze(0).transpose(0, 1)
spectrograms.append(spec)
spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
#spectrograms = spectrograms.cuda()

output = trained_model(spectrograms)
output = F.log_softmax(output, dim=2)
output = output.transpose(0, 1)

decoded_preds = GreedyDecoder(output.transpose(0, 1))
our_prediction = decoded_preds[0]

firebaseConfig = {
    "apiKey": "AIzaSyDhzMUuSY78CsILoSNZoaZN0H2WHpBUTjM",
    "authDomain": "speech-to-text-converter-aa732.firebaseapp.com",
    "projectId": "speech-to-text-converter-aa732",
    "storageBucket": "speech-to-text-converter-aa732.appspot.com",
    "messagingSenderId": "520769964374",
    "appId": "1:520769964374:web:538b3e3808d9bc71f86ef7",
    "measurementId": "G-Q8VM293K57",
    "databaseURL": ""
}

app = Flask(__name__)

@app.route('/predict',methods=['GET'])
def predict():

    firebase =  pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()
    storage.child("test.wav").download("download.wav")

    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "download.wav")
    r = sr.Recognizer()
    predicted_text_english = ""
    predicted_text_urdu = ""
    
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    try:
        predicted_text_urdu = r.recognize_google(audio, language="ur")
        predicted_text_english = r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error; {0}".format(e))

    output = {
        "Predicted Text" : {
            "English" : predicted_text_english,
            "Urdu" : predicted_text_urdu,
            "Our Prediction" : our_prediction,
        },
        "Gender" : "Male"
    }
    return output

if __name__ == "__main__":
    app.run(debug=True)
