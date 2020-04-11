import os
import torch
import torchvision.transforms as transforms
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
    def __init__(self, mat_anno, data_dir, transform=None):
        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0] -1

        if self.transform:
            image = self.transform(image)
        y = torch.LongTensor([car_class])
        return image, y

def get_loader(is_training:bool, batch_size:int):
  from torch.utils.data import Dataset, DataLoader

  transforms = _get_simclr_pipeline_transform()

  if is_training:
      mat_anno = "/content/devkit/cars_train_annos.mat"
  else:
      mat_anno = "/content/devkit/cars_test_annos.mat"
  dataset = CarsDataset(mat_anno, data_dir="/content/car_ims", transform=transforms)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)