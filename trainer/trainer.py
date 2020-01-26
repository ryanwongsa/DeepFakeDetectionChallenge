from dataloader.video_sequence_dataset import VideoSequenceDataset
from feature_detectors.face_detectors.facenet.face_model import FaceModel, get_normalised_sequences, get_samples, get_image
from torchvision import transforms
import torch
from models.efficientnet.net import Net
from logger.callbacks import Callbacks
from torch.utils.data import DataLoader
from logger.checkpointer_saver import load_checkpoint

from augmentations.augment import base_aug
from tqdm import tqdm as tqdm

import torch

class Trainer(object):
    def __init__(self):
        
        self.sequence_length = 1 # should be 1 if training only single image classifier
        self.num_sequences = 10
        self.batch_size = 4
        self.num_workers = 0
        
        self.device = 'cuda'
        
        self.keep_top_k = 2
        self.face_thresholds = [0.6, 0.7, 0.7]
        self.threshold_prob = 0.99
        
        self.image_size = 128
        self.margin_factor = 0.75
        
        self.num_samples = 16
        self.isSequenceClassifier = False
        
        
        self.network_name = 'efficientnet-b0'
        self.epochs = 3
        self.save_dir = "test_saving/save_name6"
        self.checkpoint_dir = "test_saving/save_name5-16"
        self.grad_acc_num = 1
        self.lr = 0.0003
        
        
        self.train_dir = "../../deepfake-detection-challenge/train_sample_videos"
        self.train_meta_file = "../../deepfake-detection-challenge/train_sample_videos/metadata.json"

        self.valid_dir = "../../deepfake-detection-challenge/train_sample_videos"
        self.valid_meta_file = "../../deepfake-detection-challenge/train_sample_videos/metadata.json"
        
        self.init_train_dataloader(base_aug, length=32)
        self.init_valid_dataloader(length = 16)
        
        self.FM = FaceModel(keep_top_k=self.keep_top_k, 
             face_thresholds= self.face_thresholds, 
             threshold_prob = self.threshold_prob,
             device = self.device,
             image_size = self.image_size,
             margin_factor = self.margin_factor)
        
        self.transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.model = Net(self.network_name).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.log_loss_criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.cb = Callbacks(save_dir=self.save_dir)
        self.load_from_checkpoint(self.checkpoint_dir)
        
    def load_from_checkpoint(self, checkpoint_dir):
        if self.checkpoint_dir is not None:
            dict_checkpoints = load_checkpoint(self.model, self.optimizer, self.cb, self.checkpoint_dir)
            self.model = dict_checkpoints["model"]
            self.optimizer = dict_checkpoints["optimizer"]
            self.cb = dict_checkpoints["callbacks"]
            del dict_checkpoints
        
    def train(self):
        self.cb.on_start()
        for epoch in range(self.cb.epoch, self.epochs):
            self.optimizer.zero_grad()
            self.cb.on_train_start()
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                self.cb.on_train_batch_start()
                loss = self.training_step(batch, i)
                loss.backward()

                if (i+1)%self.grad_acc_num == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.cb.on_train_batch_end()
            self.cb.on_train_end()

            self.cb.on_valid_start()
            self.model.eval()
            for j, batch in enumerate(tqdm(self.valid_dataloader)): 
                self.cb.on_valid_batch_start()
                _ = self.valid_step(batch, j)
                self.cb.on_valid_batch_end()
            self.cb.on_valid_end({"model":self.model,"optimizer":self.optimizer, "callbacks":self.cb})
        self.cb.on_end()
    
    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_sequences, batch_video_labels = self.FM.extract_face_sequence_labels(batch, self.sequence_length)
            batch_sequences = torch.cat(batch_sequences,0)
            batch_video_labels = torch.cat(batch_video_labels,0)
            batch_sequences, batch_video_labels = get_samples(batch_sequences, batch_video_labels, num_samples=self.num_samples)
            batch_sequences = get_normalised_sequences(batch_sequences, self.transform, self.isSequenceClassifier)

        batch_predicted = self.model(batch_sequences)
        loss = self.criterion(batch_predicted, batch_video_labels)

        loss_item = loss.item()
        self.cb.logger.update_metric(loss_item, "train_batch_loss")
        self.cb.logger.increment_metric(loss_item, "train_mean_loss")

        return loss


    def valid_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_sequences, batch_video_labels = self.FM.extract_face_sequence_labels(batch, self.sequence_length)

            for idx, (sequences, labels) in enumerate(zip(batch_sequences, batch_video_labels)):
                sequences = get_normalised_sequences(sequences, self.transform, self.isSequenceClassifier)

                if len(sequences) == 0:
                    predicted = torch.tensor([[0.5]]).to(self.device)
                    labels = torch.tensor([[1.0]]).to(self.device)
                else:
                    predicted = self.model(sequences)
                loss = self.criterion(predicted, labels)

                log_loss = self.log_loss_criterion(predicted.mean(axis=0), labels[0])

                loss_item = loss.item()
                log_loss_item = log_loss.item()
                self.cb.logger.increment_metric(loss_item, "valid_batch_loss")
                self.cb.logger.increment_metric(loss_item, "valid_mean_loss")
                self.cb.logger.increment_metric(log_loss_item, "valid_log_loss")

        return {}
    
    def init_train_dataloader(self, aug=None, length = None):
        
        self.train_dataset = VideoSequenceDataset(
                       self.train_dir, 
                       self.train_meta_file, 
                       transform=aug, 
                       isBalanced=True, 
                       num_sequences=self.num_sequences, 
                       sequence_length=self.sequence_length, 
                       select_type="random", 
                       isValid=False
                    )
        if length is not None:
            self.train_dataset.length = length
        
        self.train_dataloader = DataLoader(self.train_dataset,
            batch_size= self.batch_size,
            shuffle= True, 
            num_workers= self.num_workers, 
            collate_fn= self.train_dataset.collate_fn,
            pin_memory= True, 
            drop_last = True,
            worker_init_fn=self.train_dataset.init_workers_fn
        )
    
    def init_valid_dataloader(self, length = None):
        self.valid_dataset = VideoSequenceDataset(
                       self.valid_dir, 
                       self.valid_meta_file, 
                       transform=None, 
                       isBalanced=False, 
                       num_sequences=self.num_sequences, 
                       sequence_length=self.sequence_length, 
                       select_type="ordered", 
                       isValid=True
                    )
        if length is not None:
            self.valid_dataset.length = length
            
        self.valid_dataloader = DataLoader(self.valid_dataset,
            batch_size= self.batch_size,
            shuffle= False, 
            num_workers= self.num_workers, 
            collate_fn= self.valid_dataset.collate_fn,
            pin_memory= True, 
            drop_last = False,
            worker_init_fn=self.valid_dataset.init_workers_fn
        )