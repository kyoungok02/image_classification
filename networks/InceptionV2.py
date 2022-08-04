# model
import torch
import torch.nn as nn
import torch.nn.functional as F
                
# define Inception class
class InceptionV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=True):
        super(InceptionV2,self).__init__()
        self.in_channels = in_channels
        self.conv_1 = conv_block(in_channels,64,kernel_size=(7,7),stride=2,padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1) 
        self.conv_2 = conv_block(64,192, kernel_size=(3,3), stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3,3),stride=2, padding=1)
        self.inception_3a = Inception_block_ModuleA(192,64,64,64,64,96,32)
        self.inception_3b = Inception_block_ModuleA(256,64,64,96,64,96,64)
        self.inception_3c = Inception_block_ModuleD(320,128,160,64,96)
        self.inception_4a = Inception_block_ModuleB(576,224,64,96,96,128,128)
        self.inception_4b = Inception_block_ModuleB(576,192,96,128,96,128,128)
        self.inception_4c = Inception_block_ModuleB(576,160,128,160,128,128,128)
        self.inception_4d = Inception_block_ModuleB(576,96,128,192,160,160,128)
        self.inception_4e = Inception_block_ModuleD(576,128,192,192,256)
        self.inception_5a = Inception_block_ModuleC(1024,352,192,160,160,112,128)
        self.inception_5b = Inception_block_ModuleC(1024,352,192,160,192,112,128)
        self.maxpool = nn.MaxPool2d(kernel_size=(7,7),stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Sequential(
                nn.Linear(1024,num_classes),
                nn.Softmax()
        )
        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        step_1 = self.maxpool_1(self.conv_1(x))
        step_1 = self.maxpool_2(self.conv_2(step_1))
        step_2 = self.inception_3a(step_1)
        step_2 = self.inception_3b(step_2)
        step_2 = self.inception_3c(step_2)
        step_3 = self.inception_4a(step_2)
        step_3 = self.inception_4b(step_3)
        step_3 = self.inception_4c(step_3)
        step_3 = self.inception_4d(step_3)
        step_3 = self.inception_4e(step_3)
        step_4 = self.inception_5a(step_3)
        step_4 = self.inception_5b(step_4)
        step_4 = self.maxpool(step_4)
        out = self.dropout(step_4)
        out = torch.flatten(out,start_dim=1)
        out = self.fc(out)
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
    
class conv_factor_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_factor_block, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )
    
    def forward(self, x):
        return self.conv_layer(x)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5,5), stride=3),
            conv_block(in_channels, 128, kernel_size=(1,1)),
            nn.Conv2d(in_channels=128, out_channels=768, kernel_size=(5,5),stride=1),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(768, num_classes),
            nn.Softmax(),
        )

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        out = self.fc(x)
        return out 
    
class Inception_block_ModuleA(nn.Module):
    def __init__(self, in_channels, out_b1, red_b2, out_b2, red_b3, out_b3, out_b4):
        super(Inception_block_ModuleA, self).__init__()
        self.branch1 = conv_block(in_channels, out_b1,kernel_size=(1,1))

        self.branch2 = nn.Sequential(
                conv_block(in_channels, red_b2, kernel_size=(1,1)),
                conv_block(red_b2, out_b2, kernel_size=(3,3),padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_b3, kernel_size=(1,1)),
            conv_block(red_b3, out_b3, kernel_size=(3,3),padding=1),
            conv_block(out_b3, out_b3, kernel_size=(3,3),padding=1),
        )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1),
                conv_block(in_channels,out_b4,kernel_size=(1,1)),
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어줍니다. 
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out

class Inception_block_ModuleB(nn.Module):
    def __init__(self, in_channels, out_b1, red_b2, out_b2, red_b3, out_b3, out_b4):
        super(Inception_block_ModuleB, self).__init__()
        self.branch1 = conv_block(in_channels, out_b1,kernel_size=(1,1))

        self.branch2 = nn.Sequential(
                conv_block(in_channels, red_b2, kernel_size=(1,1)),
                conv_factor_block(red_b2, red_b2, kernel_size=(1,3),padding=(0,1)),
                conv_factor_block(red_b2, out_b2, kernel_size=(3,1),padding=(1,0)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_b3, kernel_size=(1,1)),
            conv_factor_block(red_b3, red_b3, kernel_size=(1,3),padding=(0,1)),
            conv_factor_block(red_b3, red_b3, kernel_size=(3,1),padding=(1,0)),
            conv_factor_block(red_b3, red_b3, kernel_size=(1,3),padding=(0,1)),
            conv_factor_block(red_b3, out_b3, kernel_size=(3,1),padding=(1,0)),
        )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1),
                conv_block(in_channels,out_b4,kernel_size=(1,1)),
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어줍니다. 
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out

class Inception_block_ModuleC(nn.Module):
    def __init__(self, in_channels, out_b1, red_b2, out_b2, red_b3, out_b3, out_b4):
        super(Inception_block_ModuleC, self).__init__()
        self.branch1 = conv_block(in_channels, out_b1,kernel_size=(1,1))

        self.branch2_conv1 = conv_block(in_channels, red_b2, kernel_size=(1,1))
        self.branch2_conv2a = conv_factor_block(red_b2, out_b2, kernel_size=(1,3),padding=(0,1))
        self.branch2_conv2b = conv_factor_block(red_b2, out_b2, kernel_size=(3,1),padding=(1,0))
        
        self.branch3_conv1 = conv_block(in_channels, red_b3, kernel_size=(1,1))
        self.branch3_conv2 = conv_block(red_b3, out_b3, kernel_size=(3,3),stride=1,padding=1)
        self.branch3_conv3a = conv_factor_block(out_b3, out_b3, kernel_size=(1,3),padding=(0,1))
        self.branch3_conv3b = conv_factor_block(out_b3, out_b3, kernel_size=(3,1),padding=(1,0))

        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1),
                conv_block(in_channels,out_b4,kernel_size=(1,1)),
        )

    def forward(self, x):
        x2 = self.branch2_conv1(x)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out2 = torch.cat([self.branch2_conv2a(x2),self.branch2_conv2b(x2)],dim=1)
        out3 = torch.cat([self.branch3_conv3a(x3),self.branch3_conv3b(x3)],dim=1)
        out = torch.cat([self.branch1(x), out2, out3, self.branch4(x)], dim=1)
        return out

class Inception_block_ModuleD(nn.Module):
    def __init__(self, in_channels, red_b1, out_b1, red_b2, out_b2):
        super(Inception_block_ModuleD, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels, red_b1,kernel_size=(1,1)),
            conv_block(red_b1, out_b1,kernel_size=(3,3),stride=2,padding=1),
        )

        self.branch2 = nn.Sequential(
                conv_block(in_channels, red_b2, kernel_size=(1,1)),
                conv_block(red_b2, red_b2, kernel_size=(3,3),stride=1,padding=1),
                conv_block(red_b2, out_b2, kernel_size=(3,3),stride=2,padding=1),
        )
        
        self.branch3 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return out
