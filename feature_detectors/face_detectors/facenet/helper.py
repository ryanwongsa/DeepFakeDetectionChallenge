import torch
import numpy as np
import os
from collections.abc import Iterable
from torchvision.ops.boxes import batched_nms, nms
import torch.nn.functional as F

def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device, is_half):

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize          # TODO: find out what m is?
    minl = min(h, w)
    minl = minl * m

    # First stage
    # Create scale pyramid
    total_boxes_all = [[] for i in range(batch_size)]
    scale = m
    while minl >= 12:
        hs = int(h * scale + 1)
        ws = int(w * scale + 1)
        im_data = imresample(imgs, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        for b_i in range(batch_size):
            boxes = generateBoundingBox(reg[b_i], probs[b_i, 1], scale, threshold[0], is_half)
            # inter-scale nms
            pick = nms(boxes[:,0:4], boxes[:,4], 0.5)
            if boxes.shape[0] > 0 and pick.shape[0] > 0:
                boxes = boxes[pick, :]
                total_boxes_all[b_i].append(boxes)

        scale = scale * factor
        minl = minl * factor

    for index_i, boxes_i in enumerate(total_boxes_all):
        if len(boxes_i)>0:
            boxes_tensor_i = torch.cat(boxes_i)
        else:
            boxes_tensor_i = torch.zeros(0,9)
        total_boxes_all[index_i] = boxes_tensor_i

    batch_boxes = []
    batch_points = []
    for img, total_boxes in zip(imgs, total_boxes_all):
        points = torch.zeros(2,5,0)
        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = nms(total_boxes[:,0:4], total_boxes[:,4], 0.7)
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = torch.stack([qq1, qq2, qq3, qq4, total_boxes[:, 4]],dim=0).T
            
            total_boxes = rerec(total_boxes)
            total_boxes[:, 0:4] = total_boxes[:, 0:4].round()

            y, ey, x, ex = pad(total_boxes, w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (24, 24)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = rnet(im_data)
            out0 = out[0].T
            out1 = out[1].T

            score = out1[1, :]
            ipass = torch.where(score > threshold[1])
            
            total_boxes = torch.cat(
                [total_boxes[ipass[0], 0:4].clone(), score[ipass].clone().unsqueeze(1)], dim=1
            )
            
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes[:,0:4], total_boxes[:,4], 0.7)
                total_boxes = total_boxes[pick, :]
                
                total_boxes = bbreg(total_boxes.clone(), mv[:, pick].T)
                total_boxes = rerec(total_boxes.clone())

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            # total_boxes = np.fix(total_boxes).astype(np.int32)
            total_boxes = total_boxes.round()
            y, ey, x, ex = pad(total_boxes.clone(), w, h)

            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (48, 48)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = onet(im_data)

            out0 =out[0].T
            out1 = out[1].T
            out2 = out[2].T
            score = out2[1, :]
            points = out1
            ipass = torch.where(score > threshold[2])
            points = points[:, ipass[0]]
            
            total_boxes = torch.cat(
                [total_boxes[ipass[0], 0:4].clone(), score[ipass].clone().unsqueeze(1)], dim=1
            )
            mv = out0[:, ipass[0]]

            w_i = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h_i = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points_x = (
                w_i.repeat(5, 1) * points[0:5, :] + total_boxes[:, 0].repeat(5, 1) - 1
            )
            points_y = (
                h_i.repeat(5, 1) * points[5:10, :] + total_boxes[:, 1].repeat(5, 1) - 1
            )
            points = torch.stack([points_x, points_y], dim=0)
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv.T)
                pick = nms(total_boxes[:,0:4],total_boxes[:,4], 0.7)
                total_boxes = total_boxes[pick, :]
                points = points[:, :, pick]

        batch_boxes.append(total_boxes)
        batch_points.append(points.T)
    return batch_boxes, batch_points


def bbreg_original(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = reg.reshape((reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = torch.stack([b1, b2, b3, b4],dim=0).T
    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh, is_half):
    stride = 2
    cellsize = 12

    mask = probs >= thresh
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask.nonzero()
    if is_half:
        bb = bb.half().flip(1)
    else:
        bb = bb.float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox


def nms_original(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def pad_original(total_boxes, w, h):
    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    x[np.where(x < 1)] = 1
    y[np.where(y < 1)] = 1
    ex[np.where(ex > w)] = w
    ey[np.where(ey > h)] = h

    return y, ey, x, ex

def pad(total_boxes, w, h):
    x = total_boxes[:, 0].clone().type(torch.LongTensor)
    y = total_boxes[:, 1].clone().type(torch.LongTensor)
    ex = total_boxes[:, 2].clone().type(torch.LongTensor)
    ey = total_boxes[:, 3].clone().type(torch.LongTensor)

    x[torch.where(x < 1)] = 1
    y[torch.where(y < 1)] = 1
    ex[torch.where(ex > w)] = w
    ey[torch.where(ey > h)] = h

    return y, ey, x, ex

def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + (l.repeat(2, 1)).T
    return bboxA

def rerec_original(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


def imresample(img, sz):
    out_shape = (sz[0], sz[1])
    im_data = torch.nn.functional.interpolate(img, size=out_shape, mode="area")
    return im_data


def extract_face(imgs, box, margin=0):
    box = box.round().int()

    box_h_min = max(box[1] - margin // 2, 0)
    box_w_min = max(box[0] - margin // 2, 0)
    box_h_max = min(box[3] + margin // 2, imgs.shape[2])
    box_w_max = min(box[2] + margin // 2, imgs.shape[3])
    faces = imgs[:,:, box_h_min:box_h_max,box_w_min:box_w_max]
    
    return faces, [box_h_min, box_w_min, box_h_max, box_w_max]

def standardise_img(imgs, image_size):
    imgs = F.interpolate(imgs, size=(image_size,image_size), mode="area")
    return imgs