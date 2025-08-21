import torch.nn as nn
import torch
from torchinfo import summary
from torchstat import stat

class ECAAttention(nn.Module):
    def __init__(self, ker_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.kernel_size = nn.Parameter(torch.tensor(ker_size, dtype=torch.float32), requires_grad=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=int(self.kernel_size.item()), padding='same', groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        kernel_size = int(self.kernel_size.item())
        self.conv.kernel_size = kernel_size # Update convolution kernel size
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

