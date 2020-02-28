import torch

def audio_aug(spectrum):
    std = 0.1
    mean = 0.0
    spectrum = spectrum + torch.randn(spectrum.size()) * std + mean
    return spectrum


def more_audio_aug(spectrum):
    std = 0.3
    mean = 0.0
    spectrum = spectrum + torch.randn(spectrum.size()) * std + mean
    return spectrum