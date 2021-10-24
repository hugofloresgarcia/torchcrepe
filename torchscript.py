import torch
import torchcrepe
import librosa
import matplotlib.pyplot as plt
from torch import nn
import functools

from torchcrepe import SAMPLE_RATE, resample

# Load audio
print(f'sample rate is {SAMPLE_RATE}')
audio, sr = librosa.load('singing.mp3')
audio = torch.tensor(audio).unsqueeze(0)
audio = resample(audio, sr)

# Select a model capacity--one of "tiny" or "full"
model_size = 'tiny'

torchcrepe.load.model('cpu', model_size)

from copy import deepcopy
model = deepcopy(torchcrepe.infer.model)

class CREPEWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model = model

    def forward(self, x: torch.Tensor):

        with torch.no_grad():
            sample_rate: int = 16000
            hop_length: int = 512
            fmin: float = 50.
            fmax: float = 1046.5
            batch_size: int = 512
        
            results = []
            generator = torchcrepe.preprocess(x, sample_rate, hop_length, batch_size)

            for frames in generator:
                probabilities = self.model(frames)

                probabilities = probabilities.reshape(
                    x.size(0), -1, torchcrepe.PITCH_BINS).transpose(1, 2)

                result = torchcrepe.postprocess(probabilities, fmin, fmax, 
                                    'weighted_argmax', False)
                results.append(result)
            
        return torch.cat(results, 1)


wrapper = CREPEWrapper(model)

traced = torch.jit.script(wrapper)
# traced = torch.jit.trace(wrapper, torch.randn(1, 4800), check_inputs=[torch.randn(1, 48000)])

plt.plot(wrapper.forward(audio).squeeze(0))
plt.savefig('pitch')

plt.plot(traced(audio).squeeze(0))
plt.savefig('pitchts')
