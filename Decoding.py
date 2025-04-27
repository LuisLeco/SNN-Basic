import torch
import snntorch.spikegen as spikegen


class Decoder:
   def __init__(self, num_steps, **kwargs):
       self.num_steps = num_steps


   def decode(self, spk_rec):
       raise NotImplementedError


class RateDecoder(Decoder):
   def __init__(self, num_steps, scale=1.0):
       super().__init__(num_steps)
       self.scale = scale


   def decode(self, spk_rec):
       return spk_rec.sum(dim=0) * self.scale




class LatencyDecoder(Decoder):
   def __init__(self, num_steps, target_time=0.5, sensitivity=1.0):
       super().__init__(num_steps)
       self.target_time = target_time
       self.sensitivity = sensitivity


   def decode(self, spk_rec):
       spike_times = torch.argmax(spk_rec, dim=0).float()
       normalized_times = spike_times / (self.num_steps - 1)
       time_diff = torch.abs(normalized_times - self.target_time)
       scores = torch.exp(-self.sensitivity * time_diff)


       return scores


class FirstSpikeDecoder(Decoder):
   def __init__(self, num_steps, threshold=0.1):
       super().__init__(num_steps)
       self.threshold = threshold


   def decode(self, spk_rec):
       spike_times = (spk_rec > self.threshold).float().argmax(dim=0)
       spike_times[(spk_rec <= self.threshold).all(dim=0)] = self.num_steps
       return torch.nn.functional.one_hot(
           spike_times.argmin(dim=-1),
           num_classes=spk_rec.shape[-1]
       ).float()
