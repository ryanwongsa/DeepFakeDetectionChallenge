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
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from models.efficientnet.net import Net
from feature_detectors.face_detectors.facenet.face_detect import MTCNN
from dataloader.video_dataset import VideoDataset

from lightning.helper import *

class LightningSystem(pl.LightningModule):
    def __init__(self):
        super(LightningSystem, self).__init__()
        
        # -------------PARAMETERS--------------
        wandb_project_name = "test-project"
        
        # model parameters
        network_name = 'efficientnet-b0'
        
        # face detection parameters
        face_img_size = 128
        face_keep_all = False
        face_thresholds = [0.6, 0.7, 0.7]
        face_select_largest = True
        face_margin = 10
        
        # dataloader parameters
        self.bs = 8
        self.num_workers = 4
        self.num_frames = 10
        
        self.train_root_dir = "/dltraining/datasets"
        self.train_metadata_file = "/dltraining/datasets/train_metadata.json"
        self.isBalanced = True
        
        self.val_root_dir = "/dltraining/datasets"
        self.val_metadata_file = "/dltraining/datasets/valid_metadata.json"
        
        self.test_root_dir = "/dltraining/datasets/test_videos"
        
        # training parameters
        self.num_training_face_samples = 32
        self.lr = 0.0003
        # -------------PARAMETERS--------------       
        
        self.face_img_size = face_img_size
        self.model = Net(network_name)
        
        device = torch.device('cuda' if self.on_gpu else 'cpu')
        
        self.fd_model = MTCNN(image_size=self.face_img_size, keep_all=face_keep_all, device=device,thresholds=face_thresholds, select_largest=face_select_largest, margin=face_margin)
        self.fd_model.eval()
        
        self.criterion = nn.BCELoss()
        self.log_loss = nn.BCELoss()

        self.transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        if HAS_WANDB:
            wandb.init(project=wandb_project_name, sync_tensorboard=True)
            wandb.watch(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        
        videos_faces, videos_labels = detect_faces_for_videos(self.fd_model, self.face_img_size, videos, labels)
        
        videos_faces, videos_labels = get_samples(videos_faces, videos_labels,self.num_training_face_samples)
        videos_faces = transform_batch(videos_faces, self.transform)

        predicted = self.forward(videos_faces).squeeze()
        loss = self.criterion(predicted, videos_labels)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        
        total_loss = 0.0
        total_logloss = 0.0
        for video, label in zip(videos, labels):
            video_label = label[0]

            faces, face_labels = detect_video_faces(self.fd_model, self.face_img_size, video, label)
            if len(faces)>0:
                faces = transform_batch(faces, self.transform)
                predictions = self.forward(faces).squeeze()
                loss = self.criterion(predictions, face_labels)
                logloss = self.log_loss(predictions.mean(), video_label)
            else:
                logloss = 0.7
                loss = 0.7

            total_logloss += logloss 
            total_loss += loss
            
        avg_loss = total_loss / len(videos)
        avg_logloss = total_logloss / len(videos)
        return {'val_loss': avg_loss, 'logloss':avg_logloss}

    def validation_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logloss = torch.tensor([x['logloss'] for x in outputs]).mean()
        return {'val_loss': loss, 'val_logloss': logloss, 'log': {'val_loss': loss, 'val_logloss':logloss}, 'progress_bar': {'val_loss': loss, 'val_logloss':logloss}}

    def test_step(self, batch, batch_idx):
        source_filenames, videos = batch

        list_submission = []
        for source_filename, video in zip(source_filenames, videos):
            faces, _ = detect_video_faces(self.df_model, self.face_img_size, video)
            if len(faces)>0:
                faces = transform_batch(faces, self.transform)
                predictions = self.forward(faces).squeeze()
                if self.on_gpu:
                    predictions = float(predictions.mean().cpu().detach().numpy())
                else:
                    predictions = float(predictions.mean().detach().numpy())
            else:
                predictions = 0.5

            dict_solution = {
              "filename":source_filename,
              "label": predictions
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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = VideoDataset(self.train_root_dir, self.train_metadata_file, isBalanced=self.isBalanced, num_frames=self.num_frames)
        train_dataloader = DataLoader(train_dataset,
                batch_size= self.bs,
                shuffle= False, 
                num_workers= self.num_workers, 
                collate_fn= train_dataset.collate_fn,
                pin_memory= True, 
                drop_last = True,
                worker_init_fn=train_dataset.init_workers_fn
            )
        return train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = VideoDataset(self.val_root_dir, self.val_metadata_file, num_frames=self.num_frames)
        val_dataloader = DataLoader(val_dataset,
                batch_size= self.bs,
                shuffle= False, 
                num_workers= self.num_workers, 
                collate_fn= val_dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=val_dataset.init_workers_fn
            )
        return val_dataloader
    
    @pl.data_loader
    def test_dataloader(self):
        dataset = VideoDataset(self.test_root_dir, None, num_frames=self.num_frames)
        dataloader = DataLoader(dataset,
                batch_size= self.bs,
                shuffle= False, 
                num_workers= self.num_workers, 
                collate_fn= dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=dataset.init_workers_fn
            )
        return dataloader