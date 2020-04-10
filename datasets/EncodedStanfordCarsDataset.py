import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io

class EncodedStanfordCarsDataset(Dataset):
    def __init__(self, mat_loc, encodings_loc):
        self.full_data_set = scipy.io.loadmat(mat_loc)
        self.car_annotations = self.full_data_set['annotations'][0]
        self.encodings = torch.load(encodings_loc)


    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
      encoded = self.encodings[idx]
      car_class = self.car_annotations[idx][-2][0][0] -1
      y = torch.LongTensor([car_class])
      return encoded, y

def get_loader(is_training:bool, batch_size:int, cross_encodings:bool=False):
  if cross_encodings:
    if is_training:
      mat_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/cars_train_annos.mat"
      enc_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/stanfordCars-camodel-train_encodings.pt"
    else:
      mat_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/cars_test_annos_withlabels.mat"
      enc_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/stanfordCars-camodel-valid_encodings.pt"
  else:
    if is_training:
      mat_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/cars_train_annos.mat"
      enc_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/stanfordCars-train_encodings.pt"
    else:
      mat_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/cars_test_annos_withlabels.mat"
      enc_loc = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/stanfordCars-valid_encodings.pt"

  dataset =  EncodedStanfordCarsDataset(mat_loc, enc_loc)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)