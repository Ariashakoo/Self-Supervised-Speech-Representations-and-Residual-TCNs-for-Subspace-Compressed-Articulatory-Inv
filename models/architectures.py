import torch.nn as nn

class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1, kernel_size=3):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(self.act(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)

class A2A_TCN_v2(nn.Module):
    def __init__(self, input_dim=1024, output_dim=12, hidden_dim=256, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualDilatedConvBlock(channels=hidden_dim, dilation=d, dropout=dropout, kernel_size=3)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x.transpose(1, 2)