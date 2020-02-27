import torch
torch.backends.cudnn.benchmark = True

from trainer.base_audio_trainer import BaseAudioTrainer

from logger.new_callbacks import Callbacks
from torch.utils.data import DataLoader
from dataloader.audio_dataset import AudioDataset

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.audio_models.model_dcase import ConvNet
from models.audio_models.model_m1 import Classifier_M2, Classifier_M3
from models.audio_models.model_m0 import Classifier
from utils.mixup import *
import numpy as np

import cProfile
try:
    from apex import amp
except:
    pass
try:
    import wandb
except:
    pass

class AudioTrainer(BaseAudioTrainer):
    def __init__(self, hparams, train_length=None, valid_length=None):
        
        self.mixup = hparams.mixup
        self.cutmix = hparams.cutmix
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.train_dir = hparams.train_dir
        self.train_meta_file = hparams.train_meta_file
        self.valid_dir = hparams.valid_dir
        self.valid_meta_file = hparams.valid_meta_file
                
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
        self.pos_weight_factor = hparams.pos_weight_factor
        self.cb = Callbacks(log_every=10, save_dir=self.save_dir)
        
        self.init_train_dataloader(length=train_length)
        self.init_valid_dataloader(length = valid_length)
        
        self.init_criterion()
        
        self.init_model()
        self.set_tuning_parameters()
        self.init_optimizer()
        self.init_scheduler()

        if hparams.project_name is not None:
            self.cb.init_wandb(hparams.project_name, hparams, hparams.run_name)
            wandb.watch(self.model)

        
        if torch.cuda.device_count() > 1 and self.device == 'cuda':
            print("Using Multiple GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count())) 
        self.model.to(self.device)

        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        
        self.load_checkpoint(self.checkpoint_dir, is_model_only=self.load_model_only)

    def init_criterion(self):
        # self.criterion_name
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight_factor))
        self.log_loss_criterion = torch.nn.BCELoss() 
        self.valid_criterion = torch.nn.BCELoss()
        
    def init_model(self):
        # self.network_name
        
        model_dict = {
            "m0": Classifier,
            "m2": Classifier_M2,
            "m3": Classifier_M3,
            "dcase": ConvNet,
        }
        self.model = model_dict[self.network_name](num_classes=1)
    
    def set_tuning_parameters(self):
        # self.tuning_type
        if self.tuning_type=="freeze_bn":
            self.model.freeze_bn = True
            self.model.freeze_bn_affine = True
    
    def init_optimizer(self, lr=None):
        # self.optimizer_name
        if lr is not None:
            self.lr = lr
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=False)
    
    def init_scheduler(self):
        # self.scheduler_name
        if self.scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        else:
            self.scheduler = None
        
    '''
    1.1.1. batch process
    '''
    def batch_process(self, batch, index=None, isTraining=True):
        self.cb.on_batch_process_start()
        source_filenames, x_batch, y_batch, video_original_filenames = batch
        y_batch = y_batch.float()
        if isTraining:
            if self.mixup or self.cutmix:
                if self.mixup and (not self.cutmix):
                    x_batch, y_batch_a, y_batch_b, lam = mixup_data(x_batch, y_batch)
                elif self.cutmix and (not self.mixup):
                    x_batch, y_batch_a, y_batch_b, lam = cutmix_data(x_batch, y_batch, device=self.device)
                else:
                    x_batch, y_batch_a, y_batch_b, lam = cutmix_data(x_batch, y_batch, device=self.device) if np.random.rand() > 0.5 else mixup_data(x_batch, y_batch, device=self.device)
                y_batch_b = y_batch_b.unsqueeze(1)
                y_batch_a = y_batch_a.unsqueeze(1)
                self.cb.on_batch_process_end()
                return x_batch, y_batch_a, y_batch_b, lam
            else:
                y_batch = y_batch.unsqueeze(1)
                self.cb.on_batch_process_end()
                return x_batch, y_batch
        else:
            y_batch = y_batch.unsqueeze(1)
            self.cb.on_batch_process_end()
            return x_batch, y_batch

    '''
    1.1.2. batch train
    '''
    def batch_train_step(self, batch, index):
        self.cb.on_batch_train_step_start()
        if self.mixup or self.cutmix:
            x_batch, y_batch_a, y_batch_b, lam = batch
            preds = self.model(x_batch.to(self.device))
            loss = mixup_criterion(self.criterion, preds, y_batch_a.to(self.device), y_batch_b.to(self.device), lam)
        else:
            x_batch, y_batch = batch
            preds = self.model(x_batch.to(self.device))
            loss = self.criterion(preds, y_batch.to(self.device))
        
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
            for idx, (x_batch, y_batch) in enumerate(zip(*batch)):
                x_batch = x_batch.unsqueeze(0)
                y_batch = y_batch.unsqueeze(0)
                predicted = self.model(x_batch.to(self.device))
                loss_original = self.criterion(predicted, y_batch.to(self.device))
                predicted2 = torch.sigmoid(predicted)
                predicted2[predicted2<0.5] = 0.5
                loss = self.valid_criterion(predicted2, y_batch.to(self.device))
                predicted3 = torch.sigmoid(predicted)
                log_loss = self.log_loss_criterion(predicted3, y_batch.to(self.device))
                
                self.cb.on_batch_valid_step_end({"valid_batch_loss":loss.item(), "valid_log_loss": log_loss.item(), "valid_original_loss":loss_original.item()})
        
    def init_train_dataloader(self, length = None):
        train_dataset = AudioDataset(self.train_dir, self.train_meta_file, spec_aug=False, isBalanced=True, isValid=False)
        if length is not None:
            train_dataset.length = length
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers= self.num_workers, collate_fn= train_dataset.collate_fn, pin_memory= True,  drop_last = True, worker_init_fn=train_dataset.init_workers_fn)
        
    def init_valid_dataloader(self, length = None):
        valid_dataset = AudioDataset(self.valid_dir, self.valid_meta_file, spec_aug=False, isBalanced=False, isValid=True)
        if length is not None:
            valid_dataset.length = length
        self.validloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers= self.num_workers,pin_memory= True, collate_fn= valid_dataset.collate_fn, drop_last = False, worker_init_fn=valid_dataset.init_workers_fn)
    