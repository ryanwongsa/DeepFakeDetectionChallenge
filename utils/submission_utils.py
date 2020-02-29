import cv2
import torch
import math
import numpy as np
from pathlib import Path
from torchvision import transforms

transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def model_loader(ClassModel, network_name, model_dir):
    model = ClassModel(network_name,None)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model

def readVideoSequence(videoFile, sequence_length, num_sequences):
    cap = cv2.VideoCapture(str(videoFile))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_index = 0
    end_index = int(fcount-start_index-sequence_length)
    gap = math.ceil(end_index/num_sequences)
    selected_frames = list(range(start_index,end_index,gap))

    list_of_squences = []
    f = 0
    
    while f <fcount:
        grabbed = cap.grab()
        
        if f in selected_frames:
            try:
                ret, frame = cap.retrieve()
                frames = []
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

                sequence_range_list = list(range(sequence_length-1))

                for i in sequence_range_list:
                    grabbed = cap.grab()
                    ret, frame = cap.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    f+=1

                frames = np.array(frames)
                list_of_squences.append(torch.tensor(frames))
            except Exception as e:
                print("read video exception:", e)
                pass
        f+=1
    cap.release()
    return list_of_squences