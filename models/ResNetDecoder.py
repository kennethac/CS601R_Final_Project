import torchvision
import torch.nn as nn

from models.FeatureClassifier import FeatureClassifier

class ResNetDecoder(nn.Module):
  def __init__(self, num_features=40, single_output=False):
    super(ResNetDecoder, self).__init__()

    if single_output:
      self.decoder = FeatureClassifier(1000, num_features)
    else:
      self.decoders = [ FeatureClassifier(1000, 2) for _ in range(num_features)]
      self.decoder = lambda x: [ d(x) for d in self.decoders ]
      for i, d in enumerate(self.decoders):
        self.add_module(f"Decoder{i}", d)
  
  def forward(self, x):
    decoded = self.decoder(x)
    return decoded