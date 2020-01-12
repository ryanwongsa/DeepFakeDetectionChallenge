import os
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

# from models.baseline.net import Net
from models.efficientnet.net import Net
from feature_detectors.face_detectors.facenet.face_detect import MTCNN
from dataloader.video_dataset import VideoDataset
import pandas as pd
import numpy as np

from lightning.helper import *

class LightningSystem(pl.LightningModule):

    def __init__(self):
        super(LightningSystem, self).__init__()
        if HAS_WANDB:
            wandb.init(project="test-project", sync_tensorboard=True)
        self.face_img_size = 300
        # self.model = Net(self.face_img_size)
        self.model = Net('efficientnet-b0')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=self.face_img_size, keep_all=False, device=device,thresholds=[0.6, 0.7, 0.7], select_largest=True, margin=20)
        self.mtcnn.eval()
        if HAS_WANDB:
            wandb.watch(self.model)
        self.criterion = nn.BCELoss()
        self.log_loss = nn.BCELoss()

        self.transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def forward(self, x):
        return self.model(x)

    def detect_faces(self, video, label=None):
        faces, _ = self.mtcnn(video.float(), return_prob=True)
        indices = [i for i, vf in enumerate(faces) if vf[0] is not None]
        faces = [vf for vf in faces if vf[0] is not None]
        face_labels=None
        if len(faces)!=0:
          # faces = torch.stack(faces, dim=1).squeeze()
          faces = torch.cat(faces)
          if label is not None:
            face_labels = label[indices]
        else:
          faces = torch.zeros(0,3,self.face_img_size,self.face_img_size)
          if label is not None:
            face_labels = torch.zeros(0,1)
        return faces, face_labels

    def transform_batch(self, videos):
        videos = torch.stack([self.transform(video/255.0) for video in videos])
        return videos

    def training_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        videos_faces = []
        videos_labels = []
        for video, label in zip(videos, labels):
          faces, face_labels = self.detect_faces(video, label)
          videos_faces.append(faces)
          videos_labels.append(face_labels)

        videos_faces = torch.cat(videos_faces)
        videos_labels = torch.cat(videos_labels)

        num_frames = videos_labels.shape[0]
        #TODO: maybe in of num_samples and num_frames
        choices = get_random_sample_frames(num_frames, num_samples=16)

        videos_faces = videos_faces[choices]
        videos_labels = videos_labels[choices]

        videos_faces = self.transform_batch(videos_faces)

        y_hat = self.forward(videos_faces).squeeze()
        loss = self.criterion(y_hat, videos_labels)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        total_loss = 0.0
        total_logloss = 0.0
        for video, label in zip(videos, labels):
          video_label = label[0]
          
          faces, face_labels = self.detect_faces(video, label)
          if len(faces)>0:
            faces = self.transform_batch(faces)

            y_hat = self.forward(faces).squeeze()

            loss = self.criterion(y_hat, face_labels)
            total_loss += loss
            logloss = self.log_loss(y_hat.mean(), video_label)
            total_logloss += logloss 
          else:
            total_loss += 0.7
            total_logloss += 0.7 

        avg_loss = total_loss / len(videos)
        avg_logloss = total_logloss / len(videos)
        return {'val_loss': avg_loss, 'logloss':avg_logloss}

    def validation_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logloss = torch.tensor([x['logloss'] for x in outputs]).mean()
        return {'val_loss': loss, 'logloss': logloss, 'log': {'val_loss': loss, 'logloss':logloss}, 'progress_bar': {'val_loss': loss, 'logloss':logloss}}

    def test_step(self, batch, batch_idx):
        source_filenames, videos = batch

        list_submission = []
        for source_filename, video in zip(source_filenames,videos):
            faces, _ = self.detect_faces(video)
            if len(faces)>0:
              faces = self.transform_batch(faces)

              y_hat = self.forward(faces).squeeze()

              dict_solution = {
                  "filename":source_filename,
                  "label": float(y_hat.mean().cpu().detach().numpy())
              }
              list_submission.append(dict_solution)
            else:
              dict_solution = {
                  "filename":source_filename,
                  "label": 0.5
              }
              list_submission.append(dict_solution)
        return {'submission_batch': list_submission}

    def test_end(self, outputs):
        list_submission = []

        for output in outputs:
            list_submission += output["submission_batch"]
        df = pd.DataFrame(list_submission)
        df.to_csv("submission.csv", index=False)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.003)

    @pl.data_loader
    def train_dataloader(self):
        train_root_dir = "/content/DeepFakeDetectionChallenge/dfdc_train_part_0"
        train_metadata_file = "/content/DeepFakeDetectionChallenge/dfdc_train_part_0/metadata.json"
        train_dataset = VideoDataset(train_root_dir, train_metadata_file, isBalanced=True)
        train_dataloader = DataLoader(train_dataset,
                batch_size= 4,
                shuffle= True, 
                num_workers= 2, 
                collate_fn= train_dataset.collate_fn,
                pin_memory= True, 
                drop_last = True,
                worker_init_fn=train_dataset.init_workers_fn
            )
        return train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        val_root_dir = "/content/DeepFakeDetectionChallenge/train_sample_videos"
        val_metadata_file = "/content/DeepFakeDetectionChallenge/train_sample_videos/metadata.json"
        val_dataset = VideoDataset(val_root_dir, val_metadata_file)
        val_dataloader = DataLoader(val_dataset,
                batch_size= 8,
                shuffle= False, 
                num_workers= 2, 
                collate_fn= val_dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=val_dataset.init_workers_fn
            )
        return val_dataloader
    
    @pl.data_loader
    def test_dataloader(self):
        root_dir = "/content/DeepFakeDetectionChallenge/test_videos"
        dataset = VideoDataset(root_dir, None)
        dataloader = DataLoader(dataset,
                batch_size= 8,
                shuffle= False, 
                num_workers= 2, 
                collate_fn= dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=dataset.init_workers_fn
            )
        return dataloader