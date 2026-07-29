import torch
import torch.nn as nn

class Conv1DLSTM(nn.Module):
    def __init__(self, num_channels=12, num_classes=5, conv_channels=64, lstm_hidden_size=128, lstm_layers=2, dropout=0.3):
        super(Conv1DLSTM, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=conv_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, time_steps, channels]
        x = x.permute(0, 2, 1)  # Conv1D espera: [batch_size, channels, time_steps]
        x = self.feature_extractor(x)
        
        x = x.permute(0, 2, 1)  # LSTM espera: [batch_size, time_steps, features]
        out, _ = self.lstm(x)
        
        # Tomar la salida del último paso temporal
        out = self.classifier(out[:, -1, :])
        return out