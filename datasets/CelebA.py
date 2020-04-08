def get_dataset(is_training:bool):
  import torchvision
  import torchvision.datasets
  import torchvision.transforms as transforms

  if is_training:
    t = transforms.Compose(
          [
          transforms.RandomGrayscale(),
          transforms.RandomApply([transforms.RandomRotation(45)], p=0.1),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.ToTensor()
          ]
        )
  else:
    t = transforms.ToTensor()
  dataset = torchvision.datasets.celeba.CelebA("/content", download=True, transform=t, split="train" if is_training else "test")
  return dataset

def get_loader(is_training:bool, batch_size:int):
  from torch.utils.data import Dataset, DataLoader
  dataset = get_dataset(is_training)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)