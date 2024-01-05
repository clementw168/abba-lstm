import torch


class ABBAForecastingLSTM(torch.nn.Module):
    def __init__(self, language_size: int, hidden_size: int, num_layers: int):
        super(ABBAForecastingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(language_size, hidden_size)
        self.lstm = torch.nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, language_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        h0 = (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            .to(x.device)
            .requires_grad_()
        )
        c0 = (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            .to(x.device)
            .requires_grad_()
        )
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


class RegressionForecastingLSTM(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super(RegressionForecastingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)

        h0 = (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            .to(x.device)
            .requires_grad_()
        )
        c0 = (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            .to(x.device)
            .requires_grad_()
        )
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out.squeeze(1)
