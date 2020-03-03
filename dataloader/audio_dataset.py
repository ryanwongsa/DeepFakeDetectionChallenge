import torch
from pathlib import Path
# import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from utils_helper.audio_helpers import get_default_conf, read_as_melspectrogram, mono_to_color
from PIL import Image
from torchvision.transforms import transforms

class AudioDataset(Dataset):
    def __init__(self, root_dir, metadata_file, time_mask=0.1, freq_mask=0.1, spec_aug=True, isBalanced=False, isValid=False):
        self.root_dir = Path(root_dir)
        self.metadata = json.load(open(metadata_file,'r'))
        self.list_videos = list(self.metadata.keys())
  
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.isBalanced = isBalanced
        if isBalanced:
            self.fake_list = [key for key, val in self.metadata.items() if val['audio_label']=='FAKE']
            self.real_list = [key for key, val in self.metadata.items() if val['audio_label']!='FAKE']

        
        if self.isBalanced:
            self.length = min(len(self.fake_list), len(self.real_list))*2
        else:
            self.length = len(self.metadata)
            
        self.isValid = isValid
        
        self.conf = get_default_conf()
        
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        
        if isValid == False:
            self.transforms = transforms_dict["train"]
        else:
            self.transforms = transforms_dict["test"]
        
        self.spec_aug = spec_aug
        
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):
        source_filenames, audios, labels, video_original_filenames = zip(*samples)
        return source_filenames, torch.stack(audios,0), torch.stack(labels,0), video_original_filenames

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
        
        image = read_as_melspectrogram(self.conf, sound_filename, trim_long_data=False)
        image = mono_to_color(image)
        
        time_dim, base_dim = image.shape[1], image.shape[0]
        if self.isValid:
            crops = list(range(0,time_dim-base_dim,(time_dim-base_dim)//10))
            images = []
            for crop in crops:
                image2 = image[:, crop:crop + base_dim, ...]
                image2 = Image.fromarray(image2[...,0], mode='L')
                image2 = self.transforms(image2)
                images.append(image2)
            image = torch.stack(images,0)
        else:
            crop = np.random.randint(0, time_dim - base_dim)
            image = image[:, crop:crop + base_dim, ...]
            if self.spec_aug:
                freq_mask_begin = int(np.random.uniform(0, 1 - self.freq_mask) * base_dim)
                image[freq_mask_begin:freq_mask_begin + int(self.freq_mask * base_dim), ...] = 0
                time_mask_begin = int(np.random.uniform(0, 1 - self.time_mask) * base_dim)
                image[:, time_mask_begin:time_mask_begin + int(self.time_mask * base_dim), ...] = 0
            image = Image.fromarray(image[...,0], mode='L')
            image = self.transforms(image)
        if video_metadata["audio_label"] == 'FAKE':
            label = 1
        else:
            label = 0

        video_original_filename = video_metadata["original"] if "original" in video_metadata else ""
             
        return source_filename, image, torch.tensor(label), video_original_filename