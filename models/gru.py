import torch
import torch.nn as nn

class GRUPuritiesPredictor(nn.Module):
    def __init__(
        self,
        feature_cnt: int = 21,
        hidden_size: int = 512,
        num_layers: int = 3,
        batch_size: int = 256,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.pred_cnt = 2
        self.h0 = torch.zeros(num_layers * (1+bidirectional), batch_size, hidden_size)
        self.lstm = nn.GRU(
            input_size=feature_cnt,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(
            in_features=hidden_size * (1 + bidirectional),
            out_features=self.pred_cnt
        )
    
    def forward(self, input):
        output, _ = self.lstm(input, self.h0.detach())
        output = output[:, -1, :]
        output = self.fc(output)
        return output


