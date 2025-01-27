import torch.nn as nn
from generator import Generator

class Derainer(nn.Module):
    
    def __init__(self, device):
        super(Derainer, self).__init__()
        self.generator = Generator().to(device)
        checkpoint = torch.load('deraining_weights.pth')
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator.eval()
    
    def forward(self, x):
        return self.generator(x)