import typing 
from typing import List, Dict
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
SPLITS = ['train', 'valid']


class FlotationDataset(Dataset):
    def __init__(
        self,
        split: str,
        time_step: int = 6,
    ):
        assert split in SPLITS, f'split should be in {SPLITS}'

        self.split = split
        self.time_step = time_step

        self.df = pd.read_csv(os.path.join('data', f"{split}.csv"))
    


    def __len__(self):
        return len(self.df) - self.time_step
    
    def __getitem__(self, index):
        features = self.df.iloc[index:index+self.time_step, :-2]
        labels = self.df.iloc[index+self.time_step, -2:]
        features_tensor = torch.tensor(features.values).to(torch.float32)
        labels_tensor = torch.tensor(labels.values).to(torch.float32)
        return{
            'feature': features_tensor,
            'label': labels_tensor,
        }

    @property
    def get_time_step(self):
        return self.time_step



if __name__ == '__main__':
    dataset = FlotationDataset(split='valid', time_step=6)
    print(dataset[0]['feature'].shape)
    print(dataset[0]['label'])
    print(len(dataset))
    pass