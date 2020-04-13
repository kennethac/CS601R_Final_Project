import os
import torch
import torchvision.transforms as transforms
import torchvision.models
from torch.utils.data import Dataset
import scipy.io
from PIL import Image
from datasets.GaussianBlur import GaussianBlur

def _get_simclr_pipeline_transform():
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    input_shape = (256,256,3)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x)
                                            ])
    return data_transforms

class CarsDataset(Dataset):
    def __init__(self, mat_anno, data_dir, transform=None, encode=False):
        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.data_dir = data_dir
        self.transform = transform

        self.encode = encode
        
        if self.encode:
            self.resnet = torchvision.models.resnet50(pretrained=True)
            self.cache = { }

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        if self.encode and idx in self.cache:
            return self.cache[idx]

        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0] -1

        if self.transform:
            image = self.transform(image)
        y = torch.LongTensor([car_class])

        if self.encode:
            image = self.resnet(image)
            image = image.detach()
            self.cache[idx] = image, y

        return image, y

def get_loader(is_training:bool, batch_size:int, encode=False):
  from torch.utils.data import Dataset, DataLoader

  transforms = _get_simclr_pipeline_transform()

  if is_training:
      mat_anno = "/content/devkit/cars_train_annos.mat"
      data_dir = "/content/cars_train"
  else:
      mat_anno = "/content/gdrive/My Drive/SimCLR/data/stanfordCars/cars_test_annos_withlabels.mat"
      data_dir = "/content/cars_test"

  dataset = CarsDataset(mat_anno, data_dir=data_dir, transform=transforms, encode=encode)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)