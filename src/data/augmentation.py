import torch
from torch.utils.data import Dataset

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True, dataset_name=None):
        self.dataset = dataset
        self.hflip = hflip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        image, calib, labels, mask, dist = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image, labels, mask, dist = random_hflip(image, labels, mask, dist)

        return image, calib, labels, mask, dist

    
def random_hflip(image, labels, mask, dist):
    image = torch.flip(image, (-1,))
    labels = torch.flip(labels.int(), (-1,)).bool()
    mask = torch.flip(mask.int(), (-1,)).bool()
    dist = torch.flip(dist.int(), (-1,))
    return image, labels, mask, dist