# model
import torch
import torch.nn as nn
import torch.nn.functional as F
                
# define Inception class

class InceptionResNetV2(nn.Module):
    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, in_channels=3, num_classes=10, init_weights=True):
        super(InceptionResNetV2,self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(A):
            blocks.append(Inception_Resnet_A(384))
        blocks.append(ReductionA(384, k, l, m, n))
        for i in range(B):
            blocks.append(Inception_Resnet_B(1152))
        blocks.append(ReductionB(1152))
        for i in range(C):
            blocks.append(Inception_Resnet_C(2144))
        self.features = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # drop out
        self.dropout = nn.Dropout2d(0.2)
        self.linear = nn.Linear(2144, num_classes)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        out = torch.flatten(x,start_dim=1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)            
                
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )
    
    def forward(self, x):
        return self.conv_layer(x)

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=3, stride=2, padding=0),
            conv_block(32, 32, kernel_size=3, stride=1, padding=0),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.branch3x3_conv = conv_block(64, 96, kernel_size=3, stride=2, padding=0)
        self.branch3x3_pool = nn.MaxPool2d(4, stride=2, padding=1)
        
        self.branch7x7a = nn.Sequential(
            conv_block(160, 64, kernel_size=1, stride=1, padding=0),
            conv_block(64, 96, kernel_size=3, stride=1, padding=0),
        )
        
        self.branch7x7b = nn.Sequential(
            conv_block(160, 64, kernel_size=1, stride=1, padding=0),
            conv_block(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            conv_block(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            conv_block(64, 96, kernel_size=3, stride=1, padding=0),
        )
        
        self.branchpoola = conv_block(192, 192, kernel_size=3, stride=2, padding=0)
        self.branchpoolb = nn.MaxPool2d(4, 2, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.branch3x3_conv(x),self.branch3x3_pool(x)],dim=1)
        x = torch.cat([self.branch7x7a(x), self.branch7x7b(x)],dim=1)
        out = torch.cat([self.branchpoola(x),self.branchpoolb(x)],dim=1)
        return out
    
class Inception_Resnet_A(nn.Module):
    def __init__(self,in_channels):
        super(Inception_Resnet_A,self).__init__()
        
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1, stride=1, padding=0)
        
        self.branch3x3 = nn.Sequential(
                conv_block(in_channels, 32, kernel_size=1, stride=1, padding=0),
                conv_block(32, 32, kernel_size=3, stride=1, padding=1),
        )
        
        self.branch3x3stack = nn.Sequential(
                conv_block(in_channels, 32, kernel_size=1, stride= 1, padding=0),
                conv_block(32, 48, kernel_size=3, stride=1, padding=1),
                conv_block(48, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat([self.branch1x1(x),self.branch3x3(x),self.branch3x3stack(x)], dim=1)
        x = self.reduction1x1(x)
        x = self.bn(x_shortcut + x)
        out = self.relu(x)
        return out
    
class ReductionA(nn.Module):
    def __init__(self,in_channels, k, l, m, n):
        super(ReductionA,self).__init__()
        
        self.branchpool = nn.MaxPool2d(3,2)
        self.branch3x3 = conv_block(in_channels, n, kernel_size=3, stride=2, padding=0)
        self.branch3x3stack = nn.Sequential(
                conv_block(in_channels, k, kernel_size=1, stride=1, padding=0),
                conv_block(k, l, kernel_size=3, stride=1, padding=1),
                conv_block(l, m, kernel_size=3, stride=2, padding=0),
        )
        self.out_channel = in_channels + n + m
    
    def forward(self, x):
        out = torch.cat([self.branchpool(x), self.branch3x3(x),self.branch3x3stack(x)],dim=1)
        return out

class Inception_Resnet_B(nn.Module):
    def __init__(self,in_channels):
        super(Inception_Resnet_B,self).__init__()
        
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0)
        
        self.branch7x7 = nn.Sequential(
                conv_block(in_channels, 128, kernel_size=1, stride=1, padding=0),
                conv_block(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
                conv_block(160, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
        )
        
        self.reduction1x1 = nn.Conv2d(384, 1152, kernel_size=1, stride=1,padding=0)
        self.shortcut = nn.Conv2d(in_channels, 1152, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1152)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat([self.branch1x1(x),self.branch7x7(x)], dim=1)
        x = self.reduction1x1(x) * 0.1
        x = self.bn(x_shortcut + x)
        out = self.relu(x)
        return out
    
class ReductionB(nn.Module):
    def __init__(self,in_channels):
        super(ReductionB,self).__init__()
        
        self.branchpool = nn.MaxPool2d(3,2)
        self.branch3x3a = nn.Sequential(
                conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0),
                conv_block(256, 384, kernel_size=3, stride=2, padding=0),
        )
        self.branch3x3b = nn.Sequential(
                conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0),
                conv_block(256, 288, kernel_size=3, stride=2, padding=0),
        )
        self.branch3x3stack = nn.Sequential(
                conv_block(in_channels,256, kernel_size=1, stride=1, padding=0),
                conv_block(256, 288, kernel_size=3, stride=1, padding=1),
                conv_block(288, 320, kernel_size=3, stride=2, padding=0),
        )
    
    def forward(self, x):
        out = torch.cat([self.branchpool(x), self.branch3x3a(x), self.branch3x3b(x),self.branch3x3stack(x)],dim=1)
        return out

class Inception_Resnet_C(nn.Module):
    def __init__(self,in_channels):
        super(Inception_Resnet_C,self).__init__()
        
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0)
        
        self.branch3x3 = nn.Sequential(
                conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0),
                conv_block(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
                conv_block(224, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
        )
        
        self.reduction1x1 = nn.Conv2d(448, 2144, kernel_size=1, stride=1,padding=0)
        self.shortcut = nn.Conv2d(in_channels, 2144, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(2144)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat([self.branch1x1(x),self.branch3x3(x)], dim=1)
        x = self.reduction1x1(x) * 0.1
        x = self.bn(x_shortcut + x)
        out = self.relu(x)
        return out                
        