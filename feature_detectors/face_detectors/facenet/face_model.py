from feature_detectors.face_detectors.facenet.helper import standardise_img, extract_face
from feature_detectors.face_detectors.facenet.face_detect import MTCNN
import torch
from PIL import Image
import numpy as np

class FaceModel(object):
    def __init__(self, 
                 keep_top_k=2, 
                 face_thresholds= [0.6, 0.7, 0.7], 
                 threshold_prob = 0.99,
                 device = 'cuda',
                 
                 image_size = 128,
                 margin_factor = 0.75,
                 pnet_pth="pretrained_models/pnet.pt",
                 rnet_pth="pretrained_models/rnet.pt",
                 onet_pth="pretrained_models/onet.pt"
                ):
        self.keep_top_k = keep_top_k
        self.face_thresholds = face_thresholds
        self.threshold_prob = threshold_prob
        self.device = device

        self.image_size = image_size
        self.margin_factor = margin_factor

        self.face_detector = MTCNN(keep_top_k=keep_top_k, device=device,thresholds=face_thresholds, threshold_prob=threshold_prob, pnet_pth=pnet_pth, rnet_pth=rnet_pth, onet_pth=onet_pth)
        self.face_detector.eval();
        
    def extract_face_sequence_labels(self, batch, sequence_length, test=False):
        if test:
            ids, batch_sequences = batch
        else:
            ids, batch_sequences, batch_labels, orig_ids = batch
        batch_video_data, batch_video_labels = [], []
        for index, video_sequences in enumerate(batch_sequences):
            video_data = []
            if test==False:
                label = batch_labels[index]
            for sequence in video_sequences:
                sequence = sequence.float().to(self.device).permute(0,3,1,2)

                index_face_analysis = sequence_length//2
                # TODO: update this dynamically based on video size
                min_face_size = 20
                boxes, probs = self.face_detector(sequence[index_face_analysis].unsqueeze(0), min_face_size=min_face_size, return_prob=True)

                # getting boxes[0] since we did unsqueeze 0, might need to adjust this if face analysis on every item in sequence
                if boxes[0] is not None:
                    for box, prob in zip(boxes[0], probs[0]):
                        box_height = box[3]-box[1]
                        margin = (box_height/self.margin_factor - box_height).round().int()

                        faces, _ = extract_face(sequence, box, margin)
                        standard_faces = standardise_img(faces, self.image_size)
                        video_data.append(standard_faces)
        #                 for standard_face in standard_faces:
        #                     print(prob.item())
        #                     display(get_image(standard_face))

            if len(video_data) > 0:
                video_data = torch.stack(video_data,0)
            else:
                video_data = torch.zeros((0,sequence_length, 3, self.image_size, self.image_size)).to(self.device)
            if test==False:
                labels = (torch.ones(video_data.shape[0],1)*label).to(self.device)
                batch_video_labels.append(labels)
                
            batch_video_data.append(video_data)
#         batch_sequences = torch.cat(batch_video_data,0)
#         batch_video_labels = torch.cat(batch_video_labels,0)
        return batch_video_data, batch_video_labels

        
def get_image(img, device = 'cuda'):
    if device == 'cuda':
        img = img.permute(1,2,0).cpu().numpy().astype('uint8')
    else:
        img = img.permute(1,2,0).numpy().astype('uint8')
    return Image.fromarray(img)

def get_random_samples(num_total, num_samples):
    indices = list(range(num_total))
    choices = np.random.choice(indices, min(num_total, num_samples), replace=False)
    return choices

def get_samples(batch_sequences, batch_labels, num_samples):
    num_total = batch_sequences.shape[0]
    choices = get_random_samples(num_total, num_samples=num_samples)
    
    batch_sequences = batch_sequences[choices]
    batch_labels = batch_labels[choices]
    
    return batch_sequences, batch_labels

def get_normalised_sequences(sample_sequences, transform, isSequenceClassifier, batch_video_labels=None):
    b, s, c, h, w = sample_sequences.shape
    sample_sequences = sample_sequences.view(b*s, c, h, w)
    sample_sequences = transform_batch(sample_sequences, transform)
    if isSequenceClassifier:
        sample_sequences = sample_sequences.view(b,s, c, h, w)
        if batch_video_labels is not None:
            batch_video_labels = batch_video_labels.repeat(1, s).unsqueeze(2)

    return sample_sequences, batch_video_labels

def transform_batch(videos, transform):
    if len(videos)>0:
        videos = torch.stack([transform(video/255.0) for video in videos])
    return videos