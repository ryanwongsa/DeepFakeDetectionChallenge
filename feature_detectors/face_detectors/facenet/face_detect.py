import torch
from torch import nn
import numpy as np
import os
from collections.abc import Iterable
from feature_detectors.face_detectors.facenet.helper import *

class PNet(nn.Module):
    def __init__(self, pretrained=True, pretrained_path='pretrained_models/pnet.pt'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            state_dict_path = pretrained_path
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    def __init__(self, pretrained=True, pretrained_path='pretrained_models/rnet.pt'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = pretrained_path
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    def __init__(self, pretrained=True, pretrained_path= 'pretrained_models/onet.pt'):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = pretrained_path
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):

    def __init__(
        self, thresholds=[0.6, 0.7, 0.7], factor=0.709,
        select_largest=True, keep_top_k=1, device=None, threshold_prob = 0.9,
        pnet_pth='pretrained_models/pnet.pt', rnet_pth='pretrained_models/rnet.pt', onet_pth='pretrained_models/onet.pt',
        is_half=True
    ):
        super().__init__()

        self.thresholds = thresholds
        self.factor = factor
        self.keep_top_k = keep_top_k
        self.threshold_prob = threshold_prob

        self.pnet = PNet(pretrained_path=pnet_pth)
        self.rnet = RNet(pretrained_path=rnet_pth)
        self.onet = ONet(pretrained_path=onet_pth)

        self.device = device
        self.is_half = is_half


    def forward(self, img, min_face_size=20, return_prob=False):
        batch_boxes, batch_probs = self.detect(img, min_face_size)
        return batch_boxes, batch_probs
        

    def detect(self, img, min_face_size=10):

        with torch.no_grad():
            batch_boxes = detect_face(
                img, min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device,
                self.is_half
            )

        boxes, probs = [], []
        for box in batch_boxes:
            box_new = box[box[:,4]>self.threshold_prob]
            lowered_threshold = False
            if len(box_new)==0:
                box_new = box[box[:,4]>0.7]
                lowered_threshold = True
            box = box_new    
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
            else:
                
                _, box_order = torch.sort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), descending=True)
                if lowered_threshold == False:
                    keep = min(len(box_order), self.keep_top_k)
                else:
                    keep = min(len(box_order), 1)
                box_order = box_order[0:keep]
                
                box_i = box[box_order][:,0:4]
                prob_i = box[box_order][:, 4]
                boxes.append(box_i)
                probs.append(prob_i)

        return boxes, probs