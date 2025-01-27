from torchvision import transforms
from ultralytics import YOLO

class Detector:
    
    def __init__(self, device):
        self.yolo = YOLO('detector.pt').to(device)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.device = device
    
    def __call__(self, x):
        x = self.transform(x)
        result = self.yolo(x.to(self.device))
        boxes = result[0].boxes.cpu().numpy()
        return boxes