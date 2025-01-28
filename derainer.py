import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from generator import Generator

class Derainer(nn.Module):
    
    def __init__(self, device="cpu"):
        super(Derainer, self).__init__()
        self.generator = Generator()
        checkpoint = torch.load('deraining_weights.pth', map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        self.device = device
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def forward(self, x):
        x = self.transform(x).unsqueeze(0).to(self.device)
        x = self.generator(x)
        x = F.interpolate(x, size=(256, 256))
        x = x.squeeze(0).detach().cpu().numpy()
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)
        x = transforms.ToPILImage()(x)
        return cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)