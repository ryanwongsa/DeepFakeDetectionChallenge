import torch
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

class BaseAudioTrainer(object):
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
        if self.scheduler:
            self.scheduler.step()
        self.cb.on_train_dl_end()

    '''
    1.1 Train Batch Main
    '''
    def batch_train(self, batch, index):
        self.cb.on_batch_train_start()
        batch = self.batch_process(batch, index, isTraining=True)
        loss = self.batch_train_step(batch, index)
        self.loss_backpass(loss)
        if (index+1)%self.grad_acc_num == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

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