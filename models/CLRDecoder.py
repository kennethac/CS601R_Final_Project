import torch.nn

class CLRNonLinearDecoder(nn.Module):
  def __init__(self, num_features=40, single_output=True):
    super(CLRDecoder, self).__init__()
    self.adapter = nn.Sequential(
        nn.ReLU(),
        nn.Linear(2048, 1000),
        nn.ReLU()
    )
    self.decoder = models.ResNetDecoder.ResNetDecoder(single_output=single_output)
    self.net = nn.Sequential(self.adapter, self.decoder)
  
  def forward(self, x):
    return self.net(x)

class CLRDecoder(nn.Module):
  def __init__(self, num_features=40, single_output=True):
    super(CLRDecoder, self).__init__()

    if single_output:
      self.net = nn.Linear(2048, num_features)
    else:
      decoders = [ nn.Linear(2048, 2) for _ in range(num_features)]
      self.net = lambda x: [ d(x) for d in decoders ]
  
  def forward(self, x):
    return self.net(x)