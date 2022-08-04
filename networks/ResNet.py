# model
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion),
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU6()
        
        if stride != 1 or in_channels != (BasicBlock.expansion * out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )
    
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        out = self.relu(x)
        return out

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channels, out_channels,stride=1):
        super(BottleNeck,self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels,out_channels*BottleNeck.expansion, kernel_size=(1,1),stride=1,bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion),
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU6()
        
        if stride !=1 or in_channels != out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=(1,1),stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        out = self.relu(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, in_channels, num_classes,init_weights=True):
        super(ResNet,self).__init__()
        
        self.start_channel = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=(7,7),stride=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1),
        )
        self.conv2x = self._make_layer(block,64,num_block[0],1)
        self.conv3x = self._make_layer(block,128,num_block[1],2)
        self.conv4x = self._make_layer(block,256,num_block[2],2)
        self.conv5x = self._make_layer(block,512,num_block[3],2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
        if init_weights:
            self._initialize_weights()

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
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.start_channel, out_channels, stride))
            self.start_channel = out_channels * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        step1 = self.conv1(x)
        step2 = self.conv2x(step1)
        step2 = self.conv3x(step2)
        step2 = self.conv4x(step2)
        step2 = self.conv5x(step2)
        out = self.avgpool(step2)
        out = torch.flatten(out,start_dim=1)
        out = self.fc(out)
        return out

def resnet18(in_channels,num_classes,init_weights):
    return ResNet(BasicBlock, [2,2,2,2],in_channels,num_classes,init_weights)

def resnet34(in_channels,num_classes,init_weights):
    return ResNet(BasicBlock,[3,4,6,3],in_channels,num_classes,init_weights)

def resnet50(in_channels,num_classes,init_weights):
    return ResNet(BottleNeck, [3,4,6,3],in_channels,num_classes,init_weights)

def resnet101(in_channels,num_classes,init_weights):
    return ResNet(BottleNeck,[3,4,23,3],in_channels,num_classes,init_weights)

def resnet152(in_channels,num_classes,init_weights):
    return ResNet(BottleNeck,[3,8,36,3],in_channels,num_classes,init_weights)
        

            
            
        