from enum import Enum

class LoaderType(Enum):
  CelebA = 1
  Stanford = 5
  StanfordCross = 6

def get_loader(is_training:bool, batch_size:int, loader_type:LoaderType):
  if LoaderType == LoaderType.CelebA:
    from datasets.EncodedCelebADataset import get_loader as gl
    return gl(is_training, batch_size)
  else:
    from datasets.EncodedStanfordCarsDataset import get_loader as gl
    return gl(is_training, batch_size, cross_encodings=loader_type == LoaderType.StanfordCross)