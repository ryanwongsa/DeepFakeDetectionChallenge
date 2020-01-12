import numpy as np
import torch

def get_random_sample_frames(num_frames, num_samples):
    indices = list(range(num_frames))
    choices = np.random.choice(indices, min(num_samples, num_frames), replace=False)
    return choices


def get_samples(videos_faces, videos_labels, num_training_face_samples):
    num_face_samples = videos_labels.shape[0]
    choices = get_random_sample_frames(num_face_samples, num_samples=num_training_face_samples)
    videos_faces = videos_faces[choices]
    videos_labels = videos_labels[choices]
    return videos_faces, videos_labels

def detect_video_faces(fd_model, face_img_size, video, label=None):
    faces, _ = fd_model(video.float(), return_prob=True)
    indices = [i for i, vf in enumerate(faces) if vf[0] is not None]
    faces = [vf for vf in faces if vf[0] is not None]
    face_labels=None
    if len(faces)!=0:
        faces = torch.cat(faces)
        if label is not None:
            face_labels = label[indices]
    else:
        faces = torch.zeros(0,3,face_img_size,face_img_size)
        if label is not None:
            face_labels = torch.zeros(0,1)
    return faces, face_labels

def detect_faces_for_videos(fd_model, face_img_size, videos, labels):
    videos_faces = []
    videos_labels = []
    for video, label in zip(videos, labels):
        faces, face_labels = detect_video_faces(fd_model, face_img_size, video, label)
        videos_faces.append(faces)
        videos_labels.append(face_labels)

    videos_faces = torch.cat(videos_faces)
    videos_labels = torch.cat(videos_labels)
    
    return videos_faces, videos_labels

def transform_batch(videos, transform):
    videos = torch.stack([transform(video/255.0) for video in videos])
    return videos