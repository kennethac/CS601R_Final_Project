import torchvision

from models.ResNetDecoder import ResNetDecoder

class ResNetBaselineModel(nn.Module):
  def __init__(self, single_output=False):
    super(ResNetBaselineModel, self).__init__()

    self.resnet = torchvision.models.resnet50()
    self.decoder = ResNetDecoder(single_output=single_output)
  
  def forward(self, x):
    features = self.resnet(x)
    decoded = self.decoder(features)
    return decoded