import torch
import torch.nn as nn
from dataset import FlotationDataset
from torch.utils.data import DataLoader

class GRUPuritiesPredictor(nn.Module):
    def __init__(
        self,
        feature_cnt: int = 22,
        hidden_size: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.pred_cnt = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.gru = nn.GRU(
            input_size=feature_cnt,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.fc = nn.Linear(
            in_features=hidden_size * (1 + self.bidirectional),
            out_features=self.pred_cnt
        )
    
    def forward(self, input):
        h0 = torch.zeros(self.num_layers * (1 + self.bidirectional), input.shape[0], self.hidden_size)
        # h0 = h0.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        output, _ = self.gru(input, h0.detach())
        output = output[:, -1, :]
        output = self.fc(output)
        return output

if __name__ == "__main__":
    ...
