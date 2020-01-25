import torch
from pathlib import Path
import json
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import math

import os


from albumentations import (
    JpegCompression, OneOf, Compose, HorizontalFlip
)
from albumentations.augmentations.transforms import (
    Resize
)


class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, isBalanced=False, num_sequences=20, sequence_length=1, select_type="ordered", isValid=False):
        self.root_dir = Path(root_dir)

        if metadata_file == None:
            self.list_videos = list(self.root_dir.rglob('*.mp4'))
            self.metadata = None
        else:
            self.metadata = json.load(open(metadata_file,'r'))
            self.list_videos = list(self.metadata.keys())
        
        self.isBalanced = isBalanced
        if isBalanced:
            self.fake_list = [key for key, val in self.metadata.items() if val['label']=='FAKE']
            self.real_list = [key for key, val in self.metadata.items() if val['label']!='FAKE']

        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.select_type = select_type
        self.transform = transform
        
        if self.isBalanced:
            self.length = min(len(self.fake_list), len(self.real_list))//2
        elif self.metadata is None:
            self.length = len(self.list_videos)
        else:
            self.length = len(self.metadata)
            
        self.isValid = isValid
        
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):
        if self.metadata == None:
            videos, source_filenames = zip(*samples)
            return source_filenames, videos

        source_filenames, videos, labels, video_original_filenames = zip(*samples)
        return source_filenames, videos, labels, video_original_filenames

    def __len__(self):
        return self.length
    
    def check_valid_augmentation(self, idx):
        if self.isValid:
            if idx%4==0:
                self.transform = None
            elif idx%4==1:
                def transform(height, width, mappings, p=1.0):
                    return Compose([
                        JpegCompression(quality_lower=40, quality_upper=41, p=1.0)
                    ], p=p,
                    additional_targets=mappings)
                self.transform = transform, (1, 0.0)
            elif idx%4==2:
                self.transform = None, (2, 1.0)
            elif idx%4==3:
                def transform(height, width, mappings, p=1.0):
                    return Compose([
                        Resize(height//4,width//4, interpolation=1, p=1.0)
                    ], p=p,
                    additional_targets=mappings)
                self.transform = transform, (1, 0.0)
    
    def readVideoSequence(self, videoFile, idx):
        self.check_valid_augmentation(idx)
                
        cap = cv2.VideoCapture(str(videoFile))

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.transform is not None:
            fps_rate = self.transform[1][0]
            fps_skip_prob = self.transform[1][1]
        else:
            fps_rate = 1
            fps_skip_prob = 0.0
            
        if self.select_type == "ordered":
            start_index = 0
        elif self.select_type == "random":
            start_index = random.randint(0,int(self.sequence_length*fps_rate-1))
            
        end_index = int(fcount-start_index-(self.sequence_length*fps_rate))
        gap = int(end_index//(self.num_sequences))
        selected_frames = list(range(start_index,end_index,gap))
        
        list_of_squences = []
        f = 0
        while f <fcount:
            grabbed = cap.grab()
            if f in selected_frames:
                ret, frame = cap.retrieve()
                dict_frames = {}
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dict_frames["image"] = frame
                mappings = {}
                
                fps_rand = random.random()
                if fps_rand < fps_skip_prob:
                    sequence_range_list = list(range(self.sequence_length*fps_rate-fps_rate))
                else:
                    sequence_range_list = list(range(self.sequence_length-1))
                    
                counter = 1
                for i in sequence_range_list:
                    grabbed = cap.grab()
                    if fps_rand>=fps_skip_prob or i % fps_rate == 0:
                        ret, frame = cap.retrieve()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        dict_frames[str(counter)] = frame
                        mappings[str(counter)] = "image"
                        counter +=1

                    f+=1
                    
                if self.transform is not None and self.transform[0] is not None:
                    aug = self.transform[0](height, width, mappings)
                    augmented_imgs = aug(**dict_frames)
                    frames = np.array([v for k, v in augmented_imgs.items()])
                else:
                    frames = np.array([v for k, v in dict_frames.items()])

                list_of_squences.append(frames)
            f+=1
        return list_of_squences

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

        if "data_dir" in video_metadata:
            video_filename = Path(video_metadata["data_dir"])/source_filename
        else:
            video_filename = self.root_dir/source_filename

        video = self.readVideoSequence(video_filename, idx)

        if self.metadata == None:
            return source_filename, video

        video_original_filename = video_metadata["original"] if "original" in video_metadata else None

        if video_metadata["label"] == 'FAKE':
            label = 1
        else:
            label = 0
            
        return source_filename, video, label, video_original_filename