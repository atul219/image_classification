import torch
import torch.nn as nn

# conv block code
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        #print("COnv block shape: {}".format(x.shape))
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

# Inception Block code
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size = 1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size = 1),
            conv_block(red_3x3, out_3x3, kernel_size = 3, stride = 1, padding = 1)

        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size = 1),
            conv_block(red_5x5, out_5x5, kernel_size = 5, stride = 1, padding = 2)
        
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            conv_block(in_channels, out_1x1pool, kernel_size = 1)
        
        )

    
    def forward(self, x):
        x  = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
       # print("[INFO] Shape after Icneption block {}".format(x.shape))
        
        return x

# Auxiliary outputs 

class Auxiliary(nn.Module):
    def __init__(self, in_channels, n_cls):
        super(Auxiliary, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.7)
        self.avg_pool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv = conv_block(in_channels, 128, kernel_size = 1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, n_cls)

    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class GoogleNet(nn.Module):
    def __init__(self, in_channels = 3, n_cls = 10, aux_out = True):
        super(GoogleNet, self).__init__()

        assert aux_out == True or aux_out == False
        self.aux_out = aux_out

        self.conv1 = conv_block(in_channels = in_channels,
                            out_channels= 64,
                            kernel_size = (7,7), 
                            stride = (2,2),
                            padding = (3,3))

        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = conv_block(64, 192, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3,3), stride = 2, padding = 1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.dropout = nn.Dropout(p = 0.4)
        self.fc1 = nn.Linear(1024, n_cls)

        if self.aux_out:
            self.aux1 = Auxiliary(512, n_cls)
            self.aux2 = Auxiliary(528, n_cls)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # auxiliary classifier 1
        if self.aux_out and self.training:
            aux1 = self.aux1(x)
        

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)


        # auxiliary classifier 2
        if self.aux_out and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.fc1(x)

        # auxiliary classifer 3
        if self.aux_out and self.training:
            return aux1, aux2, x
        else:
            return x

