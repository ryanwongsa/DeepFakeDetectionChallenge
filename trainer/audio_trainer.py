import torch
torch.backends.cudnn.benchmark = True

from trainer.base_trainer import BaseTrainer

from logger.new_callbacks import Callbacks
from torch.utils.data import DataLoader
from dataloader.audio_dataset import AudioDataset

from augmentations.audio_aug import audio_aug, more_audio_aug
from models.efficientnet.net import Net

from utils.schedulers import GradualWarmupScheduler

import cProfile
try:
    from apex import amp
except:
    pass
try:
    import wandb
except:
    pass

class AudioTrainer(BaseTrainer):
    def __init__(self, hparams, train_length=None, valid_length=None):
        self.is_lr_finder = False
        
        self.sequence_length = hparams.sequence_length
        self.num_sequences = hparams.num_sequences
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
        
        self.cb = Callbacks(log_every=10, save_dir=self.save_dir)
        
        self.valid_length = 5
        
        if self.load_model_only == False:
            self.init_train_dataloader(audio_aug, length=train_length)
        else:
            print("APPLYING MORE AUGMENTATION")
            self.init_train_dataloader(more_audio_aug, length=train_length)

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
        self.criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
        self.log_loss_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    
    def init_model(self):
        # self.network_name
        if "efficientnet" in self.network_name:
            self.model = Net(self.network_name)
    
    def set_tuning_parameters(self):
        # self.tuning_type
        self.grad_clip = False
        self.grad_clip_norm = False
        if self.tuning_type=="grad_clip":
            self.clip_val = 2
            print("Applying gradient clipping:", self.clip_val)
            self.grad_clip = True
        if self.tuning_type=="grad_clip_norm":
            self.clip_val = 0.5
            print("Applying gradient normal clipping:", self.clip_val)
            self.grad_clip_norm = True

        if self.tuning_type=="freeze_bn":
            self.model.freeze_bn = True
            self.model.freeze_bn_affine = True
        
        
    
    def init_optimizer(self, lr=None):
        # self.optimizer_name
        if lr is not None:
            self.lr = lr
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def init_scheduler(self):
        # self.scheduler_name
        self.initialise_before_schedule = False
        if self.scheduler_name == "warmup-with-cosine":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10*len(self.trainloader))
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=len(self.trainloader), after_scheduler=scheduler_cosine)
        elif self.scheduler_name == "warmup-with-reduce":
            scheduler_relrplat = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=100, cooldown=100, verbose=True)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=len(self.trainloader), after_scheduler=scheduler_relrplat)
            self.initialise_before_schedule = True
        else:
            self.scheduler = None
        
    '''
    1.1.1. batch process
    '''
    def batch_process(self, batch, index=None, isTraining=True):
        self.cb.on_batch_process_start()
        source_filenames, audios, labels, video_original_filenames = batch
        audios = [torch.stack((a,)*3, axis=-1).permute(0,3,1,2).to(self.device) for a in audios]
        if isTraining:
            audios = torch.stack(audios,0)
            labels = labels.unsqueeze(1).repeat(1, self.num_sequences).float()
            b, s, h, w, c = audios.shape
            audios = audios.view(b* s, h, w, c)
            labels = labels.view(labels.shape[0]*labels.shape[1]).unsqueeze(1).float().to(self.device)
        else:
            labels = [x.unsqueeze(1).to(self.device).float() for x in labels.unsqueeze(1).repeat(1, audios[0].shape[0])]
        self.cb.on_batch_process_end()
        return audios, labels

    '''
    1.1.2. batch train
    '''
    def batch_train_step(self, batch, index):
        self.cb.on_batch_train_step_start()
        
        batch_sequences, batch_video_labels = batch
        batch_predicted = self.model(batch_sequences)
        loss = self.criterion(batch_predicted, batch_video_labels)
            
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
                predicted = self.model(sequences)
                loss = self.criterion(predicted, labels)
                log_loss = self.log_loss_criterion(predicted.mean(axis=0), labels[0])
                
                self.cb.on_batch_valid_step_end({"valid_batch_loss":loss.item(), "valid_log_loss": log_loss.item()})
        
    def init_train_dataloader(self, aug=None, length = None):
        train_dataset = AudioDataset(self.train_dir,self.train_meta_file, transform=aug, isBalanced=True, num_sequences=self.num_sequences, fft_multiplier=20, sequence_length=self.sequence_length, isValid=False)
        if length is not None:
            train_dataset.length = length
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers= self.num_workers, collate_fn= train_dataset.collate_fn, pin_memory= True,  drop_last = True,worker_init_fn=train_dataset.init_workers_fn)
        
    def init_valid_dataloader(self, length = None):
        valid_dataset = AudioDataset(self.valid_dir,self.valid_meta_file, transform=None, isBalanced=False, num_sequences=self.valid_length, fft_multiplier=20, sequence_length=self.sequence_length, isValid=True)
        if length is not None:
            valid_dataset.length = length
        self.validloader = DataLoader(valid_dataset, batch_size= self.batch_size, shuffle= False, num_workers= self.num_workers, collate_fn= valid_dataset.collate_fn, pin_memory= True, drop_last = False, worker_init_fn=valid_dataset.init_workers_fn)
    