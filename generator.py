from denseblock import DenseBlock
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = DenseBlock(num_layers=4, growth=64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64+256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dense2 = DenseBlock(num_layers=6, growth=128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128+768, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dense3 = DenseBlock(num_layers=8, growth=256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256+2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dense4 = DenseBlock(num_layers=8, growth=512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512+4096, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 120, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True)
        )
        self.dense5 = DenseBlock(num_layers=6, growth=120)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(120+720, 64, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dense6 = DenseBlock(num_layers=4, growth=64)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64+256, 64, kernel_size=4, stride=1, padding=1),
            nn.Dropout(0.0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dense7 = DenseBlock(num_layers=4, growth=64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64+256, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pad = nn.ReflectionPad2d(4)  # Adjust padding to ensure output size matches input size
        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, padding=1)  # Adjust padding to ensure output size matches input size
        self.out = nn.Tanh()

    def forward(self, x):
        x0 = self.down1(x)
        db1 = self.dense1(x0)
        c1 = self.conv1(db1)
        db2 = self.dense2(c1)
        c2 = self.conv2(db2)
        db3 = self.dense3(c2)
        c3 = self.conv3(db3)
        db4 = self.dense4(c3)
        c4 = self.conv4(db4)
        u1 = self.up1(c4)
        db5 = self.dense5(u1)
        u2 = self.up2(db5)
        db6 = self.dense6(u2)
        u3 = self.up3(db6)
        db7 = self.dense7(u3)
        c5 = self.conv5(db7)
        c5 = self.pad(c5)
        c6 = self.conv6(c5)
        return self.out(c6)

