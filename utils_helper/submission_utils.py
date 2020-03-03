import cv2
import torch
import math
import numpy as np
from pathlib import Path
from torchvision import transforms
import librosa
from PIL import Image
import subprocess

transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
audio_transform = transforms.Compose([transforms.ToTensor()])

def model_loader(ClassModel, network_name, model_dir):
    model = ClassModel(network_name,None)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model

def readVideoSequence(videoFile, sequence_length, num_sequences):
    cap = cv2.VideoCapture(str(videoFile))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_index = 0
    end_index = int(fcount-start_index-sequence_length)
    gap = math.ceil(end_index/num_sequences)
    selected_frames = list(range(start_index,end_index,gap))

    list_of_squences = []
    f = 0
    
    while f <fcount:
        grabbed = cap.grab()
        
        if f in selected_frames:
            try:
                ret, frame = cap.retrieve()
                frames = []
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

                sequence_range_list = list(range(sequence_length-1))

                for i in sequence_range_list:
                    grabbed = cap.grab()
                    ret, frame = cap.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    f+=1

                frames = np.array(frames)
                list_of_squences.append(torch.tensor(frames))
            except Exception as e:
                print("read video exception:", e)
                pass
        f+=1
    cap.release()
    return list_of_squences


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    else:
        print(f"found zero length audio {pathname}")
        y = np.zeros((conf.samples,), np.float32)
    # make it unified length to conf.samples
    if len(y) > conf.samples:  # long enough
        if trim_long_data:
            y = y[0:0 + conf.samples]
    else:  # pad blank
        leny = len(y)
        padding = conf.samples - len(y)  # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def read_as_melspectrogram(conf, pathname, trim_long_data):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def process_wav(name, conf):
    image = read_as_melspectrogram(conf, name, trim_long_data=False)
    image = mono_to_color(image)

    time_dim, base_dim = image.shape[1], image.shape[0]
    crops = list(range(0,time_dim-base_dim,(time_dim-base_dim)//10))
    images = []
    for crop in crops:
        image2 = image[:, crop:crop + base_dim, ...]
        image2 = Image.fromarray(image2[...,0], mode='L')
        image2 = audio_transform(image2)
        images.append(image2)
    return torch.stack(images,0)


class conf:
    sampling_rate = 8000
    duration = 10  # sec
    hop_length = 125 * 2 
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'reflect'
    samples = sampling_rate * duration
    
def get_default_conf():
    return conf