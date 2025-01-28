import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
from facenet_pytorch import InceptionResnetV1

class Recognizer(nn.Module):
    
    def __init__(self, device="cpu", threshold=0.6):
        super(Recognizer, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(device)
        # checkpoint = torch.load('recognition_weights.pth', map_location=device)
        # self.resnet.load_state_dict(checkpoint['model_state_dict'])
        self.resnet.eval()
        self.threshold = threshold
        self.device = device
        self.transform = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
        
    def forward(self, x, y):
        x, y = self.transform(x).to(self.device), self.transform(y).to(self.device)
        x = self.resnet(x)
        y = self.resnet(y)
        similarity = F.cosine_similarity(x, y)
        same_person = similarity >= self.threshold
        return same_person, similarity
        