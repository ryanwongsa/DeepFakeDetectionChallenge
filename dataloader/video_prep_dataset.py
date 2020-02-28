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

class VideoPrepDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = Path(root_dir)

        self.metadata = json.load(open(metadata_file,'r'))
        self.list_videos = list(self.metadata.keys())
        self.transform = transform
        
        self.length = len(self.metadata)
            
        
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):

        source_filenames, videos = zip(*samples)
        return source_filenames, videos

    def __len__(self):
        return self.length
    
    def readVideoSequence(self, videoFile, idx=None, selection_threshold=0.5):
        cap = cv2.VideoCapture(str(videoFile))

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        
        list_of_frames = []
        f = 0
        while f <fcount:
            grabbed = cap.grab()
            if random.random()>selection_threshold:
                ret, frame = cap.retrieve()
                dict_frames = {}
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                aug = self.transform(height, width, None)
                augmented_img = aug(image=frame)
                list_of_frames.append(torch.tensor(augmented_img['image']))
            f+=1
        cap.release()
        return list_of_frames

    def __getitem__(self, idx):
        if type(self.list_videos[idx]) == str:
            video_filename = Path(self.list_videos[idx])
        else:
            video_filename = self.list_videos[idx]
        
        source_filename = f"{video_filename.stem}.mp4"
        video_metadata = self.metadata[source_filename]
        
        if video_metadata["label"] == 'FAKE':
            label = 1
            selection_threshold = 0.9
        else:
            selection_threshold = 0.7
            label = 0
        

        if "data_dir" in video_metadata:
            video_filename = Path(video_metadata["data_dir"])/source_filename
        else:
            video_filename = self.root_dir/source_filename

        video = self.readVideoSequence(video_filename, idx, selection_threshold)

        return source_filename, video