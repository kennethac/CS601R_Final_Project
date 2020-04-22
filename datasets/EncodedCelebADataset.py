import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets

import datasets


class SubEncodedCelebADataset(Dataset):
    def __init__(self, data_root, encodings_loc:str, is_train:bool, selected_attribute:str, exclude:bool=True, transform=None):
        super(Dataset, self).__init__()
        # import pdb; pdb.set_trace()
        splits = pd.read_csv(os.path.join(data_root, "list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pd.read_csv(os.path.join(data_root, "list_attr_celeba.txt"), delim_whitespace=True, header=1)
        split_id = 0 if is_train else 1
        print(f"Split is {split_id}")
        mask = splits[1] == split_id

        self.labels = torch.as_tensor(attr[mask].values)
        self.labels = (self.labels + 1) // 2 # changes it from -1,1 to 0,1

        print("Labels shape")
        print(self.labels.shape)
        print(f"Mask shape: {mask.shape}")

        attr_names = list(attr.columns)

        remove_me = attr_names.index(selected_attribute)
        attr = self.labels
        mask = torch.zeros_like(attr[0])
        mask[remove_me] = 1

        if exclude:
            self.selected_indices = (torch.any((attr & mask).type(torch.uint8), dim=1) == 0).nonzero()
        else:
            self.selected_indices =  torch.any((attr & mask).type(torch.uint8), dim=1).nonzero()
        
        print(f"Length: {len(self.selected_indices)}")
        self.encodings = torch.load(encodings_loc)
    
    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        translated_index = self.selected_indices[idx]
        lab = self.labels[translated_index]
        return self.encodings[translated_index], lab

class EncodedCelebADataset(Dataset):
    '''
    one of target_col_idx or target should be supplied:
        target_col_idx specifies the column index from the attributes to use as the label
        target specifies a target that spans multiple columns
    '''
    def __init__(self, data_root, split='train', target_col_idx=None, target=None, cross=False, samples_each=None):
        split_map = {
            "train": 0,
            "valid": 1,
        }

        assert split in split_map
        assert (target_col_idx or target) and not (target_col_idx and target)
        if target:
            assert target in ['hair']

        splits = pd.read_csv(os.path.join(data_root, "list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pd.read_csv(os.path.join(data_root, "list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = splits[1] == split_map[split]

        if target_col_idx:
            print(f'Using the {attr.columns[target_col_idx]} attribute as the label')
            self.labels = torch.as_tensor(attr[mask].values[:, target_col_idx])
            # self.labels = self.labels.reshape(-1, 1)
        else:
            # TODO: probably should do some argmax or something
            print(f'Using a group of columns to represent {target}')
            if target == 'hair':
                # bald, black_hair, blonde_hair, brown_hair, gray_hair, receding_hairline
                col_idxs = [4, 8, 9, 11, 17, 28]
            self.labels = torch.as_tensor(attr[mask].values[:, col_idxs])
        self.labels = (self.labels + 1) // 2 # changes it from -1,1 to 0,1
        print(self.labels.shape)
        print(f"Mask shape: {mask.shape}")
        if cross:            
            self.encodings = torch.load(os.path.join(data_root, f'celeba-scmodel-{split}_encodings.pt'))
        else:
            self.encodings = torch.load(os.path.join(data_root, f'{split}_encodings.pt'))

        if samples_each is not None:
            import pdb; pdb.set_trace()
            a = pd.DataFrame(self.labels).reset_index()
            num_to_keep = samples_each
            filtered_indices = [ a[a[i] == 1][:num_to_keep]["index"].tolist() for i in range(0, a.size(1))  ]
            filtered_indices = [ j for i in filtered_indices for j in i ]

            self.labels = self.labels[filtered_indices]
            self.encodings = self.encodings[filtered_indices]


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
      try:
        return self.encodings[i], self.labels[i]
      except:
        import pdb; pdb.set_trace()
        raise

def get_loader(is_training:bool, batch_size:int, root_dir="/content/gdrive/My Drive/SimCLR/data/celeba", cross=False, samples_each=None):
  target_cols = list(range(40)) # I'm pretty sure there should be 40 columns..,
  dataset =  EncodedCelebADataset(root_dir, split="train" if is_training else "valid", target_col_idx=target_cols, cross=cross, samples_each=samples_each)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)

def get_sub_loader(is_training:bool, batch_size:int, encodings_loc:str, selected_attribute:str, root_dir="/content/gdrive/My Drive/SimCLR/data/celeba"):
  target_cols = list(range(40)) # I'm pretty sure there should be 40 columns..,
  dataset =  SubEncodedCelebADataset(root_dir, encodings_loc, is_train=is_training, selected_attribute=selected_attribute, exclude=True, transform=transforms.ToTensor())
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)