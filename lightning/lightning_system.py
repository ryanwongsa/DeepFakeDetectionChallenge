import os
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from models.baseline.net import Net
from feature_detectors.face_detectors.facenet.face_detect import MTCNN
from dataloader.video_dataset import VideoDataset
import pandas as pd

class LightningSystem(pl.LightningModule):

    def __init__(self):
        super(LightningSystem, self).__init__()
        # wandb.init(project="test-project", sync_tensorboard=True)
        self.face_img_size =64
        self.model = Net(self.face_img_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=device,thresholds=[0.6, 0.7, 0.7])
        # wandb.watch(self.model)
        self.criterion = nn.BCELoss()
        self.log_loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def detect_faces(self, videos):
        boxes, probabilities = self.mtcnn.detect(videos.cpu().numpy())
        face_imgs = torch.zeros(len(boxes),3,self.face_img_size,self.face_img_size)
        for index, box in enumerate(boxes):
            if box is not None:
                box = box.astype('int')
                xmin = max(box[0][0],0)
                ymin = max(box[0][1],0)
                xmax = min(box[0][2],videos[index].shape[1])
                ymax = min(box[0][3],videos[index].shape[0])
                face_img = videos[index][ymin:ymax,xmin:xmax]
                face_img = F.interpolate(face_img.permute(2,0,1).type(torch.FloatTensor).unsqueeze(0), size=(self.face_img_size,self.face_img_size))[0]
            else:
                # print("No face found")
                face_img = torch.ones(3, self.face_img_size,self.face_img_size)*128
            face_imgs[index] = face_img/256.0
        return face_imgs

    def training_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        face_imgs = self.detect_faces(videos)

        y_hat = self.forward(face_imgs.cuda())
        loss = self.criterion(y_hat, torch.tensor(labels).type(torch.FloatTensor).cuda())
        tensorboard_logs = {'train_loss': loss}
        # wandb.log({"train_loss": loss})
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        face_imgs = self.detect_faces(videos)

        y_hat = self.forward(face_imgs.cuda()).detach().squeeze()

        loss = self.log_loss(y_hat, torch.tensor(labels).type(torch.FloatTensor).cuda())*y_hat.shape[0]
        return {'val_loss': loss, 'total_items': y_hat.shape[0]}

    def validation_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        loss /=torch.tensor([x['total_items'] for x in outputs]).sum()
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def test_step(self, batch, batch_idx):
        source_filenames, videos = batch
        face_imgs = self.detect_faces(videos)

        y_hat = self.forward(face_imgs.cuda()).detach().squeeze()
        predicted = y_hat.cpu().detach().numpy()
        list_submission = []
        for j in range(len(predicted)):
          dict_solution = {
              "filename":source_filenames[j],
              "label": predicted[j]
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
        return torch.optim.Adam(self.parameters(), lr=0.0003)

    @pl.data_loader
    def train_dataloader(self):
        root_dir = "/content/DeepFakeDetectionChallenge/train_sample_videos"
        metadata_file = "/content/DeepFakeDetectionChallenge/train_sample_videos/metadata.json"
        dataset = VideoDataset(root_dir, metadata_file)
        dataloader = DataLoader(dataset,
                batch_size= 2,
                shuffle= True, 
                num_workers= 2, 
                collate_fn= dataset.collate_fn,
                pin_memory= True, 
                drop_last = True,
                worker_init_fn=dataset.init_workers_fn
            )
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        root_dir = "/content/DeepFakeDetectionChallenge/train_sample_videos"
        metadata_file = "/content/DeepFakeDetectionChallenge/train_sample_videos/metadata.json"
        dataset = VideoDataset(root_dir, metadata_file)
        dataloader = DataLoader(dataset,
                batch_size= 2,
                shuffle= False, 
                num_workers= 2, 
                collate_fn= dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=dataset.init_workers_fn
            )
        return dataloader
    
    @pl.data_loader
    def test_dataloader(self):
        root_dir = "/content/DeepFakeDetectionChallenge/test_videos"
        dataset = VideoDataset(root_dir, None)
        dataloader = DataLoader(dataset,
                batch_size= 2,
                shuffle= False, 
                num_workers= 2, 
                collate_fn= dataset.collate_fn,
                pin_memory= True, 
                drop_last = False,
                worker_init_fn=dataset.init_workers_fn
            )
        return dataloader