import torch
import pickle
import os

def make_save_dir(save_dir):
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

def save_checkpoint(data_dict, save_dir, step_num=0):
    if save_dir != None:
        print("Saving checkpoint to:", save_dir+"-"+str(step_num))
        make_save_dir('/'.join(save_dir.split('/')[:-1]))
        
    if torch.cuda.device_count() > 1:
        torch.save({
            "model": data_dict["model"].module.state_dict(),
            "optimizer": data_dict["optimizer"].state_dict()
#             "scheduler": data_dict["scheduler"].state_dict(),
        }, save_dir)
    else:
        torch.save({
            "model": data_dict["model"].state_dict(),
            "optimizer": data_dict["optimizer"].state_dict()
#             "scheduler": scheduler.state_dict(),
        }, save_dir+"-"+str(step_num)+".ckpt")
        
    with open(save_dir+"-"+str(step_num)+".pkl", 'wb') as output:
        pickle.dump(data_dict["callbacks"], output, pickle.HIGHEST_PROTOCOL)

def load_checkpoint(model, optimizer, callbacks, checkpoint_dir):
    if checkpoint_dir != None:
        print("Loading from checkpoint:", checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir+".ckpt")
        model.load_state_dict(checkpoint['model'])    
        optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
        
        with open(checkpoint_dir+".pkl", 'rb') as output:
            callbacks = pickle.load(output)
    return {"model":model, "optimizer": optimizer, "callbacks":callbacks}