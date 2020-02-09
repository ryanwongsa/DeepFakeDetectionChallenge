import torch
torch.backends.cudnn.benchmark = True

from trainer.base_trainer import BaseTrainer
from dataloader.video_sequence_dataset import VideoSequenceDataset
from feature_detectors.face_detectors.facenet.face_model import FaceModel, get_normalised_sequences, get_samples, get_image
from torchvision import transforms

from models.efficientnet.net import Net
from models.vgg_net.sequence_net import SequenceNet
from logger.new_callbacks import Callbacks
from torch.utils.data import DataLoader

from augmentations.augment import base_aug

from utils.schedulers import GradualWarmupScheduler

import cProfile
try:
    from apex import amp
except:
    pass
transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Trainer(BaseTrainer):
    def __init__(self, hparams, train_length=None, valid_length=None):
        self.sequence_length = hparams.sequence_length
        self.num_sequences = hparams.num_sequences
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.train_dir = hparams.train_dir
        self.train_meta_file = hparams.train_meta_file
        self.valid_dir = hparams.valid_dir
        self.valid_meta_file = hparams.valid_meta_file
        self.keep_top_k = hparams.keep_top_k
        self.face_thresholds = [0.6, 0.7, 0.7]
        self.threshold_prob = hparams.threshold_prob
        self.image_size = hparams.image_size
        self.margin_factor = hparams.margin_factor
        self.num_samples = hparams.num_samples
        self.isSequenceClassifier = hparams.is_sequence_classifier
        self.epochs = hparams.epochs
        self.save_dir = hparams.save_dir
        self.checkpoint_dir = hparams.checkpoint_dir
        self.grad_acc_num = hparams.grad_acc_num
        self.lr = hparams.lr
        self.network_name = hparams.network_name
        self.optimizer_name = hparams.optimizer_name
        self.scheduler_name = hparams.scheduler_name
        self.project_name = hparams.project_name
        self.run_name = hparams.run_name
        self.criterion_name = hparams.criterion_name
        self.use_amp = hparams.use_amp
        self.device = hparams.device
        self.load_model_only = hparams.load_model_only
        self.tuning_type = hparams.tuning_type
        
        self.cb = Callbacks(log_every=1, save_dir=self.save_dir)
        
        self.init_train_dataloader(base_aug, length=train_length)
        self.init_valid_dataloader(length = valid_length)
        
        self.init_criterion()
        
        self.init_model()
        self.set_tuning_parameters()
        self.init_optimizer()
        self.init_scheduler()

        if hparams.project_name is not None:
            self.cb.init_wandb(hparams.project_name, hparams, hparams.run_name)
        
        if torch.cuda.device_count() > 1 and self.device == 'cuda':
            print("Using Multiple GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count())) 
        self.model.to(self.device)

        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        
        self.load_checkpoint(self.checkpoint_dir, is_model_only=self.load_model_only)

    def init_criterion(self):
        # self.criterion_name
        self.criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
        self.log_loss_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    
    def init_model(self):
        # self.network_name
        if "efficientnet" in self.network_name:
            self.model = Net(self.network_name)
            
        if self.network_name == 'sequence-vgg':
            self.model = SequenceNet()
            
        self.FM = FaceModel(keep_top_k=self.keep_top_k, face_thresholds= self.face_thresholds,  threshold_prob = self.threshold_prob, device = self.device, image_size = self.image_size, margin_factor = self.margin_factor, is_half=True)
    
    def set_tuning_parameters(self):
        # self.tuning_type
        pass
    
    def init_optimizer(self, lr=None):
        # self.optimizer_name
        if lr is not None:
            self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def init_scheduler(self):
        # self.scheduler_name
        if self.scheduler_name == "warmup-with-cosine":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs*len(self.trainloader))
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=len(self.trainloader), after_scheduler=scheduler_cosine)
        elif self.scheduler_name == "warmup-with-reduce":
            scheduler_relrplat = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.08 patience=50, cooldown=50, verbose=True)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=len(self.trainloader), after_scheduler=scheduler_relrplat)
        else:
            self.scheduler = None
        
    '''
    1.1.1. batch process
    '''
    def batch_process(self, batch, index=None, isTraining=True):
        self.cb.on_batch_process_start()

        with torch.no_grad():
            batch_sequences, batch_video_labels = self.FM.extract_face_sequence_labels(batch, self.sequence_length)
            if isTraining:
                batch_sequences = torch.cat(batch_sequences,0)
                batch_video_labels = torch.cat(batch_video_labels,0)
                batch_sequences, batch_video_labels = get_samples(batch_sequences, batch_video_labels, num_samples=self.num_samples)
                batch_sequences, _ = get_normalised_sequences(batch_sequences, transform, self.isSequenceClassifier)
        
        self.cb.on_batch_process_end()
        return batch_sequences, batch_video_labels

    '''
    1.1.2. batch train
    '''
    def batch_train_step(self, batch, index):
        self.cb.on_batch_train_step_start()
        
        batch_sequences, batch_video_labels = batch
        if batch_sequences.shape[0] != 0:
            batch_predicted = self.model(batch_sequences)
            loss = self.criterion(batch_predicted, batch_video_labels)
        else:
            loss = torch.tensor(0.6931471805599453)
            
        dict_metrics = {"train_batch_loss":loss.item()}
        if self.scheduler is not None:
            dict_metrics["lr"] = self.optimizer.param_groups[0]['lr']

        self.cb.on_batch_train_step_end(dict_metrics)
        return loss
    '''
    2.1.2. batch valid
    '''
    def batch_valid_step(self, batch, index):
        self.cb.on_batch_valid_step_start()
        with torch.no_grad():
            for idx, (sequences, labels) in enumerate(zip(*batch)):
                sequences, _ = get_normalised_sequences(sequences, transform, self.isSequenceClassifier)

                if len(sequences) == 0:
                    predicted = torch.tensor([[0.5]]).to(self.device)
                    labels = torch.tensor([[1.0]]).to(self.device)
                else:
                    predicted = self.model(sequences)
                loss = self.criterion(predicted, labels)
                log_loss = self.log_loss_criterion(predicted.mean(axis=0), labels[0])
                
                self.cb.on_batch_valid_step_end({"valid_batch_loss":loss.item(), "valid_log_loss": log_loss.item()})
        
    def init_train_dataloader(self, aug=None, length = None):
        train_dataset = VideoSequenceDataset(self.train_dir, self.train_meta_file, transform=aug, isBalanced=True, num_sequences=self.num_sequences, sequence_length=self.sequence_length, select_type="random", isValid=False)
        if length is not None:
            train_dataset.length = length
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers= self.num_workers, collate_fn= train_dataset.collate_fn, pin_memory= True,  drop_last = True,worker_init_fn=train_dataset.init_workers_fn)
        
    def init_valid_dataloader(self, length = None):
        valid_dataset = VideoSequenceDataset(self.valid_dir, self.valid_meta_file,  transform=None, isBalanced=False, num_sequences=self.num_sequences,  sequence_length=self.sequence_length, select_type="ordered", isValid=True)
        if length is not None:
            valid_dataset.length = length
        self.validloader = DataLoader(valid_dataset, batch_size= self.batch_size, shuffle= False, num_workers= self.num_workers, collate_fn= valid_dataset.collate_fn, pin_memory= True, drop_last = False, worker_init_fn=valid_dataset.init_workers_fn)
    