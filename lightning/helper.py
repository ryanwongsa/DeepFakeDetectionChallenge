import numpy as np

def get_random_sample_frames(num_frames, num_samples=16):
    indices = list(range(num_frames))
    choices = np.random.choice(indices, size, replace=False)
    return choices

