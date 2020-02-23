import torch
from pathlib import Path
# import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os

class AudioDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, isBalanced=False, num_sequences=3, fft_multiplier = 20, sequence_length=200, isValid=False):
        self.root_dir = Path(root_dir)
        self.metadata = json.load(open(metadata_file,'r'))
        self.list_videos = list(self.metadata.keys())
  
        
        self.isBalanced = isBalanced
        if isBalanced:
            self.fake_list = [key for key, val in self.metadata.items() if val['audio_label']=='FAKE']
            self.real_list = [key for key, val in self.metadata.items() if val['audio_label']!='FAKE']

        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.transform = transform
        self.fft_multiplier = fft_multiplier
        
        if self.isBalanced:
            self.length = min(len(self.fake_list), len(self.real_list))
        else:
            self.length = len(self.metadata)
            
        self.isValid = isValid
        
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):
        source_filenames, audios, labels, video_original_filenames = zip(*samples)
        return source_filenames, audios, torch.tensor(labels), video_original_filenames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.isBalanced:
            choice = np.random.randint(2)
            if choice==0:
                video_choice = np.random.randint(len(self.fake_list))
                video_filename = self.fake_list[video_choice]
            else:
                video_choice = np.random.randint(len(self.real_list))
                video_filename = self.real_list[video_choice]
            video_filename = self.root_dir/video_filename
        else:
            if type(self.list_videos[idx]) == str:
                video_filename = Path(self.list_videos[idx])
            else:
                video_filename = self.list_videos[idx]
        
        source_filename = f"{video_filename.stem}.mp4"
        video_metadata = self.metadata[source_filename]
        sound_filename = f"{video_filename.stem}.wav"
        sound_filename = self.root_dir/sound_filename
        
        wave, sr = librosa.load(sound_filename, sr=8000, mono=True)
        num_seconds = wave.shape[0]/sr
        spectrum = librosa.feature.melspectrogram(wave,
                                         sr=sr,
                                         n_fft=self.fft_multiplier*round(num_seconds),
                                         hop_length=sr//100,
                                         n_mels=64)
        spectrum = np.log(spectrum + 1e-9)
        s_mean, s_std = spectrum.mean(), spectrum.std()
        spectrum = (spectrum-s_mean) / s_std
        
        spectrum = torch.from_numpy(spectrum)
#         wave, sr = torchaudio.load(sound_filename)
#         num_seconds = wave.shape[1]/sr
#         spectrum = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=self.fft_multiplier*round(num_seconds))(wave.mean(axis=0))
#         spectrum = (spectrum + 1e-9).log()
#         s_mean, s_std = spectrum.mean(), spectrum.std()
#         spectrum = (spectrum-s_mean) / s_std
        
        if spectrum.shape[1]< self.sequence_length:
            print("This sound file is short:", sound_filename)
            repeats = int(np.ceil(self.sequence_length/spectrum.shape[1]))
            spectrum = spectrum.repeat(1,repeats)
    
        if self.transform is not None:
            spectrum = self.transform(spectrum)
        num_ff = self.sequence_length
        list_of_choices = []
        if self.isValid == False:
            for i in range(self.num_sequences):
                start_index = np.random.randint(spectrum.shape[1]-num_ff)
                list_of_choices.append(start_index)
        else:
            choice_index = (spectrum.shape[1]-num_ff)//self.num_sequences
            for i in range(self.num_sequences):
                list_of_choices.append(i*choice_index)
        
        spectrum_blocks = []
        for choice in list_of_choices:
            sp = spectrum[:,choice:choice+num_ff]
            spectrum_blocks.append(sp)
        spectrum_blocks = torch.stack(spectrum_blocks,0)
        if video_metadata["audio_label"] == 'FAKE':
            label = 1
        else:
            label = 0

        video_original_filename = video_metadata["original"] if "original" in video_metadata else None
             
        return source_filename, spectrum_blocks, label, video_original_filename