import torch
import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self,in_channels = 3, n_cls = 10):
        super(AlexNet, self).__init__()


        self.conv_feat = nn.Sequential(
                        nn.Conv2d(in_channels, 96, kernel_size = 11, stride = 4, padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 2),
                        nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 2),
                        nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
                        nn.ReLU(),
                        nn.Conv2d(384, 384, kernel_size = 3, padding = 1),
                        nn.ReLU(),
                        nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 2)
        )


        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.clf = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 *6, 4096),    
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, n_cls))

    
    def forward(self, x):
        x = self.conv_feat(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.clf(x)

        return x


# if __name__ == '__main__':
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    net = AlexNet().to(device)
#    summary(net, (3,224, 224))


