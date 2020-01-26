from logger.metric_logger import MetricLogger
from logger.checkpointer_saver import save_checkpoint

class Callbacks(object):
    def __init__(self, step=0, initial_epoch=0, project_name=None, experiment_name=None, save_dir=None):
        self.save_dir = save_dir
        self.step = step
        self.initial_step = step
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.logger = MetricLogger()

    def on_train_start(self, data_dict=None):
        self.logger.reset_metrics(["train_mean_loss"])
    
    def on_train_batch_start(self, data_dict=None):
        pass
    
    def on_train_batch_end(self, data_dict=None):
        print("train step:", self.step, self.logger.get("train_batch_loss"))
        self.step +=1
        
    def on_train_end(self, data_dict=None):
        print("== train:", self.epoch, self.step, self.logger.get("train_mean_loss"))
    
    def on_valid_start(self, data_dict=None):
        self.logger.reset_metrics(["valid_mean_loss", "valid_log_loss"])
    
    def on_valid_batch_start(self, data_dict=None):
        self.logger.reset_metrics(["valid_batch_loss"])
    
    def on_valid_batch_end(self, data_dict=None):
        print("valid step: ",self.step, self.logger.get("valid_batch_loss"))
    
    def on_valid_end(self, data_dict=None):
        print("== valid: ", self.epoch, self.step, self.logger.get("valid_mean_loss"), self.logger.get("valid_log_loss"))    
        self.epoch += 1
        save_checkpoint(data_dict, self.save_dir, self.step)
        
    def on_start(self, data_dict=None):
        pass

    def on_end(self, data_dict=None):
        pass