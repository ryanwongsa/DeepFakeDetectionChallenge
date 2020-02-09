from logger.metric_logger import MetricLogger
import wandb

class Callbacks(object):
    def __init__(self, step=0, initial_epoch=0, log_every=10, save_dir=None):
        self.save_dir = save_dir
        self.step = step
        self.initial_step = step
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.logger = MetricLogger()
        
        self.log_every = log_every
        self.has_wandb = False
    
    def init_wandb(self, project_name, hparams, run_name=None):
        if run_name is not None:
            wandb.init(project=project_name, resume=run_name, allow_val_change=True)
        else:
            wandb.init(project=project_name, allow_val_change=True)
        wandb.config.update(hparams, allow_val_change=True)
        self.has_wandb =True
    
    def on_epoch_start(self, dict_data={}):
        pass
    
    def on_epoch_end(self, dict_data={}):
        self.epoch += 1
        
    def on_train_dl_start(self, dict_data={}):
        pass
        # self.logger.reset_metrics(["train_mean_loss"])
    
    def on_train_dl_end(self, dict_data={}):
        print("TRAIN:", self.epoch, self.step, self.logger.get("train_mean_loss"))
        
    def on_batch_train_start(self, dict_data={}):
        pass
        
    def on_batch_train_end(self, dict_data={}):
        pass
    
    def on_batch_process_start(self, dict_data={}):
        pass
        
    def on_batch_process_end(self, dict_data={}):
        pass
        
    def on_batch_train_step_start(self, dict_data={}):
        pass
        
    def on_batch_train_step_end(self, dict_data={}):
        self.logger.update_metric(dict_data["train_batch_loss"], "train_batch_loss")
        self.logger.increment_metric(dict_data["train_batch_loss"], "train_mean_loss")
        dict_data["train_iter_mean_loss"] = self.logger.get("train_mean_loss")
        self.send_log(dict_data)
        self.step += 1
        
        
    def on_valid_dl_start(self, dict_data={}):
        self.logger.reset_metrics(["valid_mean_loss", "valid_log_loss"])
    
    def on_valid_dl_end(self, dict_data={}):
        print("VALID:", self.epoch, self.step, self.logger.get("valid_mean_loss"), self.logger.get("valid_log_loss"))
        self.send_log({
            "valid_mean_loss": self.logger.get("valid_mean_loss"), 
            "valid_log_loss": self.logger.get("valid_log_loss")
        }, True)
        
    def on_batch_valid_start(self, dict_data={}):
        pass
    
    def on_batch_valid_end(self, dict_data={}):
        pass
    
    def on_batch_valid_step_start(self, dict_data={}):
        pass
    
    def on_batch_valid_step_end(self, dict_data={}):
        self.logger.update_metric(dict_data["valid_batch_loss"], "valid_batch_loss")
        self.logger.increment_metric(dict_data["valid_batch_loss"], "valid_mean_loss")
        self.logger.increment_metric(dict_data["valid_log_loss"], "valid_log_loss")
    
    def send_log(self, dict_logs={}, sendAlways=False):
        if dict_logs != {}:
            if sendAlways or self.step%self.log_every == 0:
                dict_logs["custom_step"]=self.step
                if self.has_wandb:
                    wandb.log(dict_logs)
                else:
                    print(self.step, dict_logs)