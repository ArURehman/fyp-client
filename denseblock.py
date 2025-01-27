import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth = growth
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(growth * (i + 1), growth, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth),
                nn.LeakyReLU(0.2, inplace=True)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)
