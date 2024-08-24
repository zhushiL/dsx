import torch 
from torch import nn 
from torch.nn import functional as F 

class Bottle_neck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        num_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(num_channels, out_channels, kernel_size=1, stride=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # X.shape (batch_szie, channels, height, width)
        res = self.conv_res(X)
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.relu(self.bn3(self.conv3(X)))
        Y = X + res
        return Y
    
def res_block(in_channels, out_channels, num_res, block_2=False):
    block = []
    for i in range(num_res):
        if i==0 :
            if block_2 == True:
                block.append(Bottle_neck(in_channels, out_channels, stride=1))
            else:
                block.append(Bottle_neck(in_channels, out_channels, stride=2))
        else:
            block.append(Bottle_neck(out_channels, out_channels, stride=1))
    return block

class ResNet50(nn.Module):
    """ ResNet50
    
        Args:
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.block_1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block_2 = nn.Sequential(*res_block(64, 256, 3, block_2=True))
        self.block_3 = nn.Sequential(*res_block(256, 512, 4))
        self.block_4 = nn.Sequential(*res_block(512, 1024, 6))
        self.block_5 = nn.Sequential(*res_block(1024, 2048, 3))

        self.flat = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten())
        
        if num_classes == 1000:
            self.fc = nn.Linear(2048, 1000)
        else:
            assert num_classes < 1000, '类别数需小于1000'
            self.fc = nn.Sequential(
                            nn.Linear(2048, 1000),
                            nn.ReLU(),
                            nn.Linear(1000, num_classes))
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.flat(x)
        x = self.fc(x)
        return x