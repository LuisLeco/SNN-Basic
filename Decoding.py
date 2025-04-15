class Decoder:
    def decode(self, spk_rec):
        raise NotImplementedError

class RateDecoder(Decoder):
    def decode(self, spk_rec):
        return spk_rec.sum(dim=0)
