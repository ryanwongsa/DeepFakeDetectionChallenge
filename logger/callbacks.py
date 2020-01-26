from logger.metric_logger import MetricLogger
from logger.checkpointer_saver import save_checkpoint

import wandb


class Callbacks(object):
    def __init__(self, step=0, initial_epoch=0, save_dir=None):
        self.save_dir = save_dir
        self.step = step
        self.initial_step = step
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.logger = MetricLogger()
        
        self.log_every = 10
        

    def init_wandb(self, project_name, hparams, run_name=None):
        if run_name is not None:
            wandb.init(project=project_name, resume=run_name, allow_val_change=True)
        else:
            wandb.init(project=project_name, allow_val_change=True)
        wandb.config.update(hparams, allow_val_change=True)
        
    def on_train_start(self, data_dict=None):
        self.logger.reset_metrics(["train_mean_loss"])
    
    def on_train_batch_start(self, data_dict=None):
        pass
    
    def on_train_batch_end(self, data_dict=None):
#         print("train step:", self.step, self.logger.get("train_batch_loss"))
        self.send_log({"train_batch_loss": self.logger.get("train_batch_loss")})
        self.step +=1
        
    def on_train_end(self, data_dict=None):
        self.send_log({"train_mean_loss": self.logger.get("train_mean_loss")})
        print("== train:", self.epoch, self.step, self.logger.get("train_mean_loss"))
    
    def on_valid_start(self, data_dict=None):
        self.logger.reset_metrics(["valid_mean_loss", "valid_log_loss"])
    
    def on_valid_batch_start(self, data_dict=None):
        self.logger.reset_metrics(["valid_batch_loss"])
    
    def on_valid_batch_end(self, data_dict=None):
#         print("valid step: ",self.step, self.logger.get("valid_batch_loss"))
#         wandb.log({"valid_batch_loss": self.logger.get("valid_batch_loss")})
        pass
    
    def on_valid_end(self, data_dict=None):
        print("== valid: ", self.epoch, self.step, self.logger.get("valid_mean_loss"), self.logger.get("valid_log_loss"))    
        self.send_log({
            "valid_mean_loss": self.logger.get("valid_mean_loss"), 
            "valid_log_loss": self.logger.get("valid_log_loss")
        }, True)
        self.epoch += 1
        save_checkpoint(data_dict, self.save_dir, self.step)
        
    def on_start(self, data_dict=None):
        pass

    def on_end(self, data_dict=None):
        pass
    
    def send_log(self, dict_logs={}, sendAlways=False):
        if dict_logs != {}:
            if sendAlways or self.step%self.log_every == 0:
                dict_logs["custom_step"]=self.step
                wandb.log(dict_logs)
        