import torch
from utils.lr_finder import ExponentialLR, LinearLR
from tqdm.auto import tqdm

try:
    from apex import amp
    HAS_AMP = True
except Exception as e:
    HAS_AMP = False
    

class BaseTrainer(object):
    def __init__(self):
        pass
    
    def batch_process(self, index=None, isTraining=True):
        raise NotImplementedError()
    
    def batch_train_step(self, batch, index):
        raise NotImplementedError()
        
    def batch_valid_step(self, batch, index):
        raise NotImplementedError()
        
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

        batch = self.batch_process(batch, index, isTraining=True)
        loss = self.batch_train_step(batch, index)
        self.loss_backpass(loss)
        if (index+1)%self.grad_acc_num == 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        self.cb.on_batch_train_end()

    '''
    1.1.3. batch backpass
    '''
    def loss_backpass(self, loss):
        if HAS_AMP and self.use_amp:
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
    
    '''
    Use this method to ensure the model can actually overfit on one batch
    '''
    def train_on_sample(self, steps=100,batch=None):
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
    def lr_finder(self, num_iter, start_lr, end_lr, step_mode="linear", stop_factor=5):
        self.init_optimizer(lr=start_lr)
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
        return lrs, losses, min_lr_index