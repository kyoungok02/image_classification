# model
import torch
import torch.nn as nn
import torch.nn.functional as F

# define Inception class
class InceptionV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=True, aux_layer=True):
        super(InceptionV1,self).__init__()
        self.in_channels = in_channels
        self.aux_layer = aux_layer
        
        if self.aux_layer:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        
        self.conv_1 = conv_block(in_channels,64,kernel_size=(7,7),stride=2,padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1) 
        self.conv_2 = conv_block(64,192, kernel_size=(3,3), stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3,3),stride=2, padding=1)
        self.inception_3a = InceptionV1_block(192,64,96,128,16,32,32)
        self.inception_3b = InceptionV1_block(256,128,128,192,32,96,64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(3,3), stride=2,padding=1)
        self.inception_4a = InceptionV1_block(480,192,96,208,16,48,64)
        self.inception_4b = InceptionV1_block(512,160,112,224,24,64,64)
        self.inception_4c = InceptionV1_block(512,128,128,256,24,64,64)
        self.inception_4d = InceptionV1_block(512,112,144,288,32,64,64)
        self.inception_4e = InceptionV1_block(528,256,160,320,32,128,128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        self.inception_5a = InceptionV1_block(832,256,160,320,32,128,128)
        self.inception_5b = InceptionV1_block(832,384,192,384,48,128,128)
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7),stride=1)
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
        step_2 = self.maxpool_3(step_2)
        step_3 = self.inception_4a(step_2)
        step_3 = self.inception_4b(step_3)
        if self.aux_layer and self.training:
            aux1 = self.aux1(step_3)
        step_3 = self.inception_4c(step_3)
        step_3 = self.inception_4d(step_3)
        if self.aux_layer and self.training:
            aux2 = self.aux2(step_3)
        step_3 = self.inception_4e(step_3)
        step_3 = self.maxpool_4(step_3)
        step_4 = self.inception_5a(step_3)
        step_4 = self.inception_5b(step_4)
        step_4 = self.avgpool(step_4)
        out = self.dropout(step_4)
        out = torch.flatten(out,start_dim=1)
        out = self.fc(out)
        if self.aux_layer and self.training:
            return out, aux1, aux2
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
    
class naive_Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5):
        super(naive_Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1,1))
        self.branch2 = conv_block(in_channels, out_3x3, kernel_size=(3,3))
        self.branch3 = conv_block(in_channels, out_5x5, kernel_size=(5,5))
        self.branch4 = nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1)

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어줍니다. 
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x
    
class InceptionV1_block(nn.Module):
    def __init__(self, in_channels, out_b1, red_b2, out_b2, red_b3, out_b3, out_b4):
        super(InceptionV1_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_b1, kernel_size=(1,1))
        self.branch2 = nn.Sequential(
                conv_block(in_channels, red_b2, kernel_size=(1,1)),
                conv_block(red_b2, out_b2, kernel_size=(3,3),padding=1),
        )
        self.branch3 = nn.Sequential(
                conv_block(in_channels, red_b3, kernel_size=(1,1)),
                conv_block(red_b3,out_b3,kernel_size=(5,5),padding=2),
        )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3,3), stride=1,padding=1),
                conv_block(in_channels, out_b4,kernel_size=(1,1)),
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어줍니다. 
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return x

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            conv_block(in_channels, 128, kernel_size=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
            nn.Softmax(),
        )

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x 
