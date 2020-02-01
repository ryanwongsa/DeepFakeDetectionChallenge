from dataloader.video_sequence_dataset import VideoSequenceDataset
from feature_detectors.face_detectors.facenet.face_model import FaceModel, get_normalised_sequences, get_samples, get_image
from torchvision import transforms
import torch
from models.efficientnet.net import Net
from models.efficientnet_sequences.sequence_net import SequenceNet
from logger.callbacks import Callbacks
from torch.utils.data import DataLoader
from logger.checkpointer_saver import load_checkpoint

from augmentations.augment import base_aug
from tqdm import tqdm as tqdm

class Trainer(object):
    def __init__(self, hparams):
        
        self.sequence_length = hparams.sequence_length # should be 1 if training only single image classifier
        self.num_sequences = hparams.num_sequences
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        
        self.device = 'cuda'
        
        self.keep_top_k = hparams.keep_top_k
        self.face_thresholds = [0.6, 0.7, 0.7]
        self.threshold_prob = hparams.threshold_prob
        
        self.image_size = hparams.image_size
        self.margin_factor = hparams.margin_factor
        
        self.num_samples = hparams.num_samples
        self.isSequenceClassifier = hparams.is_sequence_classifier
        
        
        self.network_name = hparams.network_name
        self.epochs = hparams.epochs
        self.save_dir = hparams.save_dir
        self.checkpoint_dir = hparams.checkpoint_dir
        self.grad_acc_num = hparams.grad_acc_num
        self.lr = hparams.lr
        
        
        self.train_dir = hparams.train_dir
        self.train_meta_file = hparams.train_meta_file

        self.valid_dir = hparams.valid_dir
        self.valid_meta_file = hparams.valid_meta_file
        
        self.init_train_dataloader(base_aug, length=None)
        self.init_valid_dataloader(length = None)
        
        self.FM = FaceModel(keep_top_k=self.keep_top_k, 
             face_thresholds= self.face_thresholds, 
             threshold_prob = self.threshold_prob,
             device = self.device,
             image_size = self.image_size,
             margin_factor = self.margin_factor)
        
        self.transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.model = SequenceNet(self.network_name).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.log_loss_criterion = torch.nn.BCELoss()
        
        #### ------------------------
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'efficient_net' in name or name in ['_fc.weight', '_fc.bias']:
                    param.requires_grad = False
        params = list(self.model.rnn_decoder.parameters()) + list(self.model.lstm_fc1.parameters()) + \
                  list(self.model.lstm_fc2.parameters()) + list(self.model.lstm_bn1.parameters()) + \
                  list(self.model.lstm_bn2.parameters()) + list(self.model.lstm_fc3.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        #### ------------------------
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.cb = Callbacks(save_dir=self.save_dir)
        if hparams.project_name is not None:
            self.cb.init_wandb(hparams.project_name, hparams, hparams.run_name)
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
            batch_sequences, _ = get_normalised_sequences(batch_sequences, self.transform, self.isSequenceClassifier)
        
        
        batch_predicted = self.model(batch_sequences, model_types = [1])
        loss = self.criterion(batch_predicted, batch_video_labels)

        self.cb.logger.update_metric(loss.item(), "train_batch_loss")
        self.cb.logger.increment_metric(loss.item(), "train_mean_loss")

        return loss


    def valid_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_sequences, batch_video_labels = self.FM.extract_face_sequence_labels(batch, self.sequence_length)

            for idx, (sequences, labels) in enumerate(zip(batch_sequences, batch_video_labels)):
                sequences, _ = get_normalised_sequences(sequences, self.transform, self.isSequenceClassifier)

                if len(sequences) == 0:
                    predicted = torch.tensor([[0.5]]).to(self.device)
                    labels = torch.tensor([[1.0]]).to(self.device)
                else:
                    predicted = self.model(sequences, model_types = [1])
                loss = self.criterion(predicted, labels)
                log_loss = self.log_loss_criterion(predicted.mean(axis=0), labels[0])

                self.cb.logger.increment_metric(loss.item(), "valid_batch_loss")
                self.cb.logger.increment_metric(loss.item(), "valid_mean_loss")
                self.cb.logger.increment_metric(log_loss.item(), "valid_log_loss")

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