import torch.nn as nn
import torch.nn.functional as F
import torch
from facenet_pytorch import InceptionResnetV1

class Recognizer(nn.Module):
    
    def __init__(self, device):
        self.resnet = InceptionResnetV1().to(device)
        checkpoint = torch.load('recognition_weights.pth')
        self.resnet.load_state_dict(checkpoint['model_state_dict'])
        self.resnet.eval()
        
    def forward(self, x, y):
        x = self.resnet(x)
        y = self.resnet(y)
        distance = F.pairwise_distance(x, y)