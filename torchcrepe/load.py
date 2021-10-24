import os

import numpy as np
import torch
import torchcrepe
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, capacity='full'):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    state = torch.load(file, map_location=device)
    new_state = dict(state)
    for key in state:
        if 'conv' in key:
            num = int(key[4])
            keypat = key.replace(str(num), '')
            new_key_pattern = f'b{num}.conv'
            new_key = keypat.replace('conv', new_key_pattern)
            new_state[new_key] = state[key]
            del new_state[key]

    breakpoint()
    state = new_state


    torchcrepe.infer.model.load_state_dict(
        state, strict=True)

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()
