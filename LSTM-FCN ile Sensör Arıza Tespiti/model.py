import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMFCN(nn.Module):

    def __init__(self, input_size=590, lstm_hidden=128, num_classes=2):
        super(LSTMFCN, self).__init__()

        # LSTM: input size = 1 (single channel), time steps = input_size
        self.lstm = nn.LSTM(
            input_size=1,  # 1 feature at each time step
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # CNN: 1D Conv feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer (LSTM + CNN fusion)
        self.fc = nn.Linear(lstm_hidden + 64, num_classes)

    def forward(self, x):

        # x shape: [batch, features] -> [batch, seq_len, 1]
        x = x.unsqueeze(2)  # [B, 590] -> [B, 590, 1]

        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]  # take only the last time step

        # For CNN: [B, 590, 1] -> [B, 1, 590]
        cnn_input = x.permute(0, 2, 1)

        # 1D CNN layers
        x_cnn = F.relu(self.conv1(cnn_input))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))

        # Global average pooling: [B, 64, L] -> [B, 64, 1] -> [B, 64]
        x_cnn = self.gap(x_cnn).squeeze(2)  # [B, 64]

        # Concatenate LSTM and CNN outputs
        combined = torch.cat([lstm_feat, x_cnn], dim=1)  # [B, hidden + 64]

        # Classification
        out = self.fc(combined)  # [B, num_classes]

        return out
