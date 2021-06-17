import torch
import torch.nn as nn

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Res_block, self).__init__()

        self.identity_downsample = identity_downsample
        self.stride = stride
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = stride,
                                padding = 1)
        
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, 
                                out_channels * self.expansion,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0)

        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()

    
    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)       

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x 



class ResNet(nn.Module):
    def __init__(self, res_block, layers, img_channels, n_cls = 10):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layers(res_block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layers(res_block, layers[1], out_channels = 128, stride = 2)
        self.layer3 = self._make_layers(res_block, layers[2], out_channels = 256, stride = 2)
        self.layer4 = self._make_layers(res_block, layers[3], out_channels = 512, stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*4 , n_cls)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
    
    def _make_layers(self, res_block, num_res_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                                nn.Conv2d(self.in_channels, 
                                out_channels * 4,
                                kernel_size = 1,
                                stride = stride),
                                nn.BatchNorm2d(out_channels * 4))
        
        layers.append(res_block(self.in_channels, 
                                out_channels,
                                identity_downsample,
                                stride))
        
        self.in_channels = out_channels * 4     # 64*4 = 256

        for i in range(num_res_blocks - 1):
            layers.append(res_block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)


def ResNet50(img_channels = 3, n_cls = 10):
    return ResNet(Res_block, [3, 4, 6, 3], img_channels, n_cls)

def ResNet101(img_channels = 3, n_cls = 10):
    return ResNet(Res_block, [3, 4, 23, 3], img_channels, n_cls)

def ResNet152(img_channels = 3, n_cls = 10):
    return ResNet(Res_block, [3, 8, 36, 3], img_channels, n_cls)

