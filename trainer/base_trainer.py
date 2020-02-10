import torch
from utils.lr_finder import ExponentialLR, LinearLR
from tqdm.auto import tqdm

import pickle
import os
try:
    from apex import amp
except:
    pass
import cProfile

try:
    import wandb
except:
    pass

def make_save_dir(save_dir):
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

class BaseTrainer(object):
    def __init__(self):
        self.is_lr_finder = False
    
    def batch_process(self, index=None, isTraining=True):
        raise NotImplementedError()
    
    def batch_train_step(self, batch, index):
        raise NotImplementedError()
        
    def batch_valid_step(self, batch, index):
        raise NotImplementedError()
    
    def save_checkpoint(self, step_num, save_dir):
        if save_dir != None:
            print("Saving checkpoint to:", save_dir+"-"+str(step_num))
            make_save_dir('/'.join(save_dir.split('/')[:-1]))

            if torch.cuda.device_count() > 1:
                dict_save = {
                    "model": self.model.module.state_dict()
                }
            else:
                dict_save = {
                    "model": self.model.state_dict(),
                }
            dict_save["optimizer"] = self.optimizer.state_dict()

            if self.scheduler is not None:
                dict_save["scheduler"] = self.scheduler.state_dict()

            if self.use_amp:
                dict_save["amp"] = amp.state_dict()

            torch.save(dict_save, save_dir+"-"+str(step_num)+".ckpt")

            with open(save_dir+"-"+str(step_num)+".pkl", 'wb') as output:
                pickle.dump(self.cb, output, pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, checkpoint_dir, is_model_only = False):
        if checkpoint_dir != None:
            print("Loading from checkpoint:", checkpoint_dir)
            checkpoint = torch.load(checkpoint_dir+".ckpt")
            self.model.load_state_dict(checkpoint['model'])
            if is_model_only == False:
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                with open(checkpoint_dir+".pkl", 'rb') as output:
                    self.cb = pickle.load(output)

                if "amp" in checkpoint and self.use_amp:
                    amp.load_state_dict(checkpoint["amp"])
    
    '''
    1. Train Main
    '''
    def train_on_dl(self):
        self.cb.on_train_dl_start()
        self.model.train()
        self.optimizer.zero_grad()
        for index, batch in enumerate(tqdm(self.trainloader)):
            self.batch_train(batch, index)

        self.cb.on_train_dl_end()

    '''
    1.1 Train Batch Main
    '''
    def batch_train(self, batch, index):
        self.cb.on_batch_train_start()
#         try:
        batch = self.batch_process(batch, index, isTraining=True)
        loss = self.batch_train_step(batch, index)
        self.loss_backpass(loss)
        if self.grad_clip:
            if self.use_amp:
                torch.nn.utils.clip_grad_value_(amp.master_params(self.optimizer), self.clip_val)
            else:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_val)
        elif self.grad_clip_norm:
            if self.use_amp:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.clip_val)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
        
        
        if (index+1)%self.grad_acc_num == 0:
            self.optimizer.step()
            if self.scheduler is not None and self.is_lr_finder == False:
                if self.initialise_before_schedule and 0.4>index/len(self.trainloader):
                    self.scheduler.step(self.cb.step,self.cb.logger.get("train_mean_loss"), True)
                else:
                    self.scheduler.step(self.cb.step,self.cb.logger.get("train_mean_loss"))
            elif self.is_lr_finder:
                self.scheduler.step()
            self.optimizer.zero_grad()
#         except Exception as e:
#             print(e)
        self.cb.on_batch_train_end()

    '''
    1.1.3. batch backpass
    '''
    def loss_backpass(self, loss):
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
    '''
    2. Validation Main
    '''
    def valid_on_dl(self):
        self.cb.on_valid_dl_start()
        self.model.eval()
        with torch.no_grad():
            for index, batch in enumerate(tqdm(self.validloader)):
                self.batch_valid(batch, index)

        self.cb.on_valid_dl_end()

    '''
    2.1 Validation Batch Main
    '''
    def batch_valid(self, batch, index):
        self.cb.on_batch_valid_start()
        
        batch = self.batch_process(batch, index, isTraining=False)
        self.batch_valid_step(batch, index)

        self.cb.on_batch_valid_end()
    
    def train(self):
        for epoch in range(self.cb.epoch, self.epochs):
            self.cb.on_epoch_start()
            self.train_on_dl()
            self.valid_on_dl()
            self.cb.on_epoch_end()
            self.save_checkpoint(self.cb.step, self.save_dir)
    
    def get_batch(self):
        return next(iter(self.trainloader))
    
    '''
    Use this method to ensure the model can actually overfit on one batch
    '''
    def train_on_sample(self, steps=100, log_every=10, batch=None):
        self.cb.log_every = log_every
        if batch is None:
            batch = next(iter(self.trainloader))
            
        self.cb.on_train_dl_start()
        self.model.train()
        self.optimizer.zero_grad()
        for i in tqdm(range(steps)):
            self.batch_train(batch, i)
        self.cb.on_train_dl_end()
        
        self.cb.on_valid_dl_start()
        self.model.eval()
        with torch.no_grad():
            self.batch_valid(batch, 0)
        self.cb.on_valid_dl_end()
        print("If running in a notebook, don't forget to reset the trainer")
    
    '''
    Use this method to find good learning rates
    '''
    def lr_finder(self, num_iter, start_lr, end_lr, step_mode="linear", stop_factor=5, log_every=1):
        self.cb.log_every=log_every
        self.is_lr_finder = True
        self.init_model()
        self.set_tuning_parameters()
        self.init_optimizer(lr=start_lr)

        if torch.cuda.device_count() > 1 and self.device == 'cuda':
            print("Using Multiple GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count())) 
        self.model.to(self.device)

        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        if self.cb.has_wandb:
            wandb.watch(self.model)
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))
        iterator = iter(self.trainloader)
        self.scheduler = lr_schedule
        self.model.train()
        lrs = []
        losses = []
        min_loss = 100000
        min_lr = 1000
        counter = 0
        min_lr_index = 0
        for iteration in tqdm(range(num_iter)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.trainloader)
                batch = next(iterator)

            self.cb.on_train_dl_start()
            self.optimizer.zero_grad()
            self.batch_train(batch, iteration)
            lr = self.scheduler.get_lr()[0]
            loss = self.cb.logger.get("train_mean_loss")
            if loss < min_loss:
                min_loss = loss
                min_lr = lr
                min_lr_index = iteration
            if loss>min_loss*stop_factor:
                counter +=1
                if counter>5:
                    print(loss, min_loss)
                    print("MIN_LR (index):", min_lr, min_lr_index)
                    break
            else:
                counter = 0
            lrs.append(lr)
            losses.append(loss)
        print("If running in a notebook, don't forget to reset the trainer")
        self.is_lr_finder = False
        return lrs, losses, min_lr_index