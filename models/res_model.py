
import torch.nn as nn
import torchvision.models.resnet as resnet

conv1x1=resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, dataset="mnist"):
        super(ResNet, self).__init__()
        self.inplanes = 32 # conv1에서 나올 채널의 차원 -> 이미지넷보다 작은 데이터이므로 32로 조정

        # inputs = 3x224x224 -> 3x128x128로 바뀜
        if dataset == "mnist" or dataset == "fmnist":
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1) # 3 반복
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2) # 4 반복
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2) # 6 반복
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2) # 3 반복
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1): # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input [32, 128, 128] -> [C ,H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x.shape =[32, 64, 64]

        x = self.layer1(x)
        #x.shape =[128, 64, 64]
        x = self.layer2(x)
        #x.shape =[256, 32, 32]
        x = self.layer3(x)
        #x.shape =[512, 16, 16]
        x = self.layer4(x)
        #x.shape =[1024, 8, 8]
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    
def ResNet10(d, nc):
    return ResNet(resnet.BasicBlock, [1,1,1,1], num_classes=nc, zero_init_residual=True, dataset=d)

def ResNet14(d, nc):
    return ResNet(resnet.BasicBlock, [1,1,2,2], num_classes=nc, zero_init_residual=True, dataset=d)
    
def ResNet16(d, nc):
    return ResNet(resnet.BasicBlock, [1,2,2,2], num_classes=nc, zero_init_residual=True, dataset=d)

###########
    
def ResNet18(d, nc):
    return ResNet(resnet.BasicBlock, [2,2,2,2], num_classes=nc, zero_init_residual=True, dataset=d)

def ResNet34(d, nc):
    return ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=nc, zero_init_residual=True, dataset=d)

def ResNet50(d, nc):
    return ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=nc, zero_init_residual=True, dataset=d)

def ResNet101(d, nc):
    return ResNet(resnet.Bottleneck, [3, 4, 23, 3], num_classes=nc, zero_init_residual=True, dataset=d)

def ResNet152(d, nc):
    return ResNet(resnet.Bottleneck, [3, 8, 36, 3], num_classes=nc, zero_init_residual=True, dataset=d)