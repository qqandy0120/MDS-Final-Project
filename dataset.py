import typing 
from typing import List, Dict
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
SPLITS = ['train', 'valid']
MODES = ['normal', 'y_only']

class FlotationDataset(Dataset):
    def __init__(
        self,
        split: str,
        time_step: int = 6,
        mode: str = 'normal',
    ):
        assert split in SPLITS, f'split should be one of {SPLITS}'
        assert mode in MODES, f'mode should be one of {MODES}'

        self.split = split
        self.time_step = time_step
        self.mode = mode

        self.df = pd.read_csv(os.path.join('data', f"{split}.csv"))
    


    def __len__(self):
        return len(self.df) - self.time_step
    
    def __getitem__(self, index):
        if self.mode == 'normal':
            features = self.df.iloc[index:index+self.time_step, :]
        if self.mode == 'y_only':
            features = self.df.iloc[index:index+self.time_step, -1]
        labels = self.df.iloc[index+self.time_step, -1]
        features_tensor = torch.tensor(features.values).to(torch.float32)
        labels_tensor = torch.tensor([labels]).to(torch.float32)
        return{
            'feature': features_tensor,
            'label': labels_tensor,
        }

    @property
    def get_time_step(self):
        return self.time_step



if __name__ == '__main__':
    train = FlotationDataset(split='train',  mode='y_only')
    print(len(train))
    valid = FlotationDataset(split='valid', mode='y_only')
    print(len(valid))
    print(valid[0]['feature'])
    print(valid[0]['label'])
    pass