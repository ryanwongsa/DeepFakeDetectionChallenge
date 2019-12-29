import torch
import torchvision
from pathlib import Path
import json
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image

import os

class VideoDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = root_dir

        self.list_videos = list(Path(self.root_dir).rglob('*.mp4'))
        if metadata_file == None:
          self.metadata = None
        else:
          self.metadata = json.load(open(metadata_file,'r'))

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):
        if self.metadata == None:
          videos, source_filenames = zip(*samples)
          videos = torch.stack(videos, 0)
          return source_filenames, videos

        videos, source_filenames, labels, video_original_filenames = zip(*samples)
        labels = [1 if x=="FAKE" else 0 for x in labels]
        videos = torch.stack(videos, 0)
        return source_filenames, videos, labels, video_original_filenames

    def readVideo(self, videoFile):
        cap = cv2.VideoCapture(str(videoFile))
        cap.set(cv2.CAP_PROP_FPS, 30)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        height, width = 300, 300

        frames = torch.FloatTensor(3, fcount, height, width)
        rng_choice = random.randint(0, fcount)
        for f in range(fcount):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame.astype('uint8'))
                frame = cv2.resize(frame, (width, height)) 
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame

        return frames

    def readCenterFrame(self, videoFile):
        cap = cv2.VideoCapture(str(videoFile))
        cap.set(cv2.CAP_PROP_FPS, 30)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        height, width = 580, 580

        rng_choice = fcount//2
        for f in range(fcount):
            grabbed = cap.grab()
            if f == rng_choice:
              ret, frame = cap.retrieve()
              if ret:
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  # frame = Image.fromarray(frame.astype('uint8')[0])

                  frame = cv2.resize(frame, (width, height)) 
                  frame = torch.from_numpy(frame)
                  # frame = frame.permute(2, 0, 1)

        return frame

    def __len__(self):
        return len(self.list_videos)

    def __getitem__(self, idx):
        video_filename = self.list_videos[idx]
        source_filename = f"{video_filename.stem}.mp4"
        video = self.readCenterFrame(video_filename)

        if self.metadata == None:
          return video,  source_filename


        video_metadata = self.metadata[source_filename]
        video_original_filename = video_metadata["original"] if "original" in video_metadata else None
        return video, source_filename, video_metadata["label"], video_original_filename