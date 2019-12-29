import os
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from models.baseline.net import Net
from feature_detectors.face_detectors.facenet import MTCNN

class LightningSystem(pl.LightningModule):

    def __init__(self):
        super(LightningSystem, self).__init__()
        wandb.init(project="test-project", sync_tensorboard=True)
        face_img_size =64
        self.model = Net(face_img_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=device,thresholds=[0.6, 0.7, 0.7])
        wandb.watch(self.model)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        source_filenames, videos, labels, video_original_filenames = batch
        boxes, probabilities = self.mtcnn.detect(videos.cpu().numpy())
        face_imgs = torch.zeros(len(boxes),3,face_img_size,face_img_size)
        for index, box in enumerate(boxes):
          if box is not None:
            box = box.astype('int')
            xmin = max(box[0][0],0)
            ymin = max(box[0][1],0)
            xmax = min(box[0][2],videos[index].shape[1])
            ymax = min(box[0][3],videos[index].shape[0])
            face_img = videos[index][ymin:ymax,xmin:xmax]
            face_img = F.interpolate(face_img.permute(2,0,1).type(torch.FloatTensor).unsqueeze(0), size=(face_img_size,face_img_size))[0]
          else:
            # print("No face found")
            face_img = torch.ones(3, face_img_size,face_img_size)*128
          face_imgs[index] = face_img/256.0
        y_hat = self.forward(face_imgs.cuda())
        loss = self.criterion(y_hat, torch.tensor(labels).type(torch.FloatTensor).cuda())
        tensorboard_logs = {'train_loss': loss}
        # wandb.log({"train_loss": loss})
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0003)

    @pl.data_loader
    def train_dataloader(self):
        root_dir = "/content/dfdc_train_part_0"
        metadata_file = "/content/dfdc_train_part_0/metadata.json"
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