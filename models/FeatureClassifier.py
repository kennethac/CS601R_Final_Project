import torch.nn as nn

class FeatureClassifier(nn.Module):
  def __init__(self, n_inp, n_out):
    super(FeatureClassifier, self).__init__()
    midpoint = n_inp + ( n_out - n_inp) // 2
    self.net = nn.Sequential(
      nn.Linear(n_inp, midpoint),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(midpoint, n_out)
    )
  
  def forward(self, x):
    return self.net(x)