import snntorch.spikegen as spikegen


class Encoder:
    def __init__(self, num_steps, gain=0.5):
        self.num_steps = num_steps
        self.gain = gain

    def encode(self, data):
        raise NotImplementedError


class RateEncoder(Encoder):
    def encode(self, data):
        return spikegen.rate(data, num_steps=self.num_steps, gain=self.gain)
