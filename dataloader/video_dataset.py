import torch
import torchvision
from pathlib import Path
import json
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image
import math

import os
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, isBalanced=False, num_frames=20):
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

        self.num_frames = num_frames

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def collate_fn(self, samples):
        try:
            if self.metadata == None:
                videos, source_filenames = zip(*samples)
                videos = torch.stack(videos, 0)
                return source_filenames, videos

            videos, source_filenames, labels, video_original_filenames = zip(*samples)

            # TODO: Image Resizing here
            # for index, video in enumerate(videos):

            labels = torch.stack(labels, 0)

            videos = torch.stack(videos, 0)
            return source_filenames, videos, labels, video_original_filenames
        except Exception as e:
            print("collate exception: ", e)
            videos, source_filenames = zip(*samples)
            for i in len(videos):
                print(source_filenames[i], videos[i].shape)

    def readVideo(self, videoFile):
        video, _, _ = torchvision.io.read_video(str(videoFile),pts_unit='sec')
        max_image_dim = 1920  # TODO: resize image if the max dimension is bigger than this
        total_frames = video.shape[0]
        selected_frames = list(range(0,total_frames,math.ceil(total_frames/self.num_frames)))
        video = video[selected_frames]
        video = video.permute(0, 3, 1, 2)
        height_img = video.shape[2]
        width_img = video.shape[3]
        max_dim = max([height_img, width_img, max_image_dim])
        diff_height = (max_dim-height_img)//2
        diff_width = (max_dim-width_img)//2
        p2d = (diff_width, diff_width, diff_height, diff_height)
        video = F.pad(video, p2d, 'constant', 0)
        
        return video

    def readVideo_cv2(self, videoFile):
        max_image_dim = 1920
        try:
            cap = cv2.VideoCapture(str(videoFile))
            
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            selected_frames = list(range(0,fcount,math.ceil(fcount/self.num_frames)))
            frames = torch.FloatTensor(len(selected_frames), 3, height, width)
            counter = 0
            for f in range(fcount):
                grabbed = cap.grab()
                if f in selected_frames:
                    ret, frame = cap.retrieve()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frame = torch.from_numpy(frame)
                        frame = frame.permute(2, 0, 1)
                        frames[counter, :, :, :] = frame
                        counter += 1
            max_dim = max([height, width, max_image_dim])
            diff_height = (max_dim-height)//2
            diff_width = (max_dim-width)//2
            p2d = (diff_width, diff_width, diff_height, diff_height)
            frames = F.pad(frames, p2d, 'constant', 0)
            if frames.shape[2]!=max_image_dim or frames.shape[3]!=max_image_dim:
                print("outofbounds:", str(videoFile), (height, width))
                frames = F.interpolate(frames, size=(max_image_dim, max_image_dim))
            return frames
        except Exception as e:
            print(e)
            print(str(videoFile))
            return torch.zeros(self.num_frames, 3, max_image_dim, max_image_dim)
            

    def __len__(self):
        if self.isBalanced:
            return min(len(self.fake_list), len(self.real_list))//2
        elif self.metadata is None:
            return len(self.list_videos)
        else:
            return len(self.metadata)

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

        video = self.readVideo_cv2(video_filename)

        if self.metadata == None:
            return video, source_filename

        video_original_filename = video_metadata["original"] if "original" in video_metadata else None

        if video_metadata["label"] == 'FAKE':
            labels = torch.ones(video.shape[0])
        else:
            labels = torch.zeros(video.shape[0])

        return video, source_filename, labels, video_original_filename