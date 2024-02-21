import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone()
    elif "resnet_2d" in name.lower():
        return ResNetBackbone_2D()
    elif "resnet_attention" in name.lower():
        return ResNetBackbone_Attention()
    # elif "resnet_2d_attention" in name.lower():
    #     return ResNetBackbone_Attention_2D()
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "alexnet_2d" == name.lower():
        return AlexNetBackbone_2D()
    elif "alexnet_attention" == name.lower():
        return AlexNetBackbone_Attention()
    # elif "alexnet_2d_attention" == name.lower():
    #     return AlexNetBackbone_Attention_2D()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "alexnet_2l" == name.lower():
        return AlexNetBackbone_2level()


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim


# 这里的alexnet 使用torchvision的预制模型用于处理时频图像
# 这里的2D指的是输入数据为二维图片
class AlexNetBackbone_2D(nn.Module):
    def __init__(self):
        super(AlexNetBackbone_2D, self).__init__()
        model_alexnet = models.alexnet(pretrained=False)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim


    # 这里的2d指使用二维处理多通道的一维数据
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        self.alexnet_2d = nn.Sequential(
            nn.Conv2d(1, 16, (1, 128), padding=(0, 64)),
            nn.MaxPool2d((1, 4), stride=4),
            nn.Dropout(),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, (1, 64), padding=(0, 32)),
            nn.MaxPool2d((1, 4)),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, (1, 16), padding=(0, 8)),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, (1, 4), padding=(0, 2)),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, (1, 2), padding=(0, 1)),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(),
            nn.ReLU(inplace=True),

            nn.Flatten(1, 3),

            # nn.Softmax()
        )

    def forward(self, x):
        return self.alexnet_2d(x)

    def output_num(self):
        return 4096


class AlexNetBackbone_Attention(nn.Module):
    def __init__(self):
        super(AlexNetBackbone_Attention, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (1, 128), padding=(0, 64))
        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.droup1 = nn.Dropout()

        self.conv2 = nn.Conv2d(16, 32, (1, 64), padding=(0, 32))
        self.maxpool2 = nn.MaxPool2d((1, 4))

        self.conv3 = nn.Conv2d(32, 64, (1, 16), padding=(0, 8))
        self.maxpool3 = nn.MaxPool2d((1, 4))
        self.droup3 = nn.Dropout()

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.conv4 = nn.Conv2d(64, 128, (2, 4), padding=(0, 2))
        self.maxpool4 = nn.MaxPool2d((1, 2))

        self.conv5 = nn.Conv2d(128, 256, (1, 2), padding=(0, 1))
        self.maxpool5 = nn.MaxPool2d((1, 2))
        self.droup5 = nn.Dropout()

        self.flatten = nn.Flatten(1, 3)

            # nn.Softmax()

    def forward(self, x):
        x = F.relu(self.droup1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.maxpool2(self.conv2(x)))
        x = F.relu(self.droup3(self.maxpool3(self.conv3(x))))
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = F.relu(self.maxpool4(self.conv4(x)))
        x = F.relu(self.droup5(self.maxpool5(self.conv5(x))))
        out = self.flatten(x)
        return out

    def output_num(self):
        return 4096

# 这里Restnet不是标准的restnet18、50结构，而是根据Alexnet添加残差块的结构，用于同等的对比
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 128+1), padding=(0, 64))
        self.maxpool = nn.MaxPool2d((1, 4), stride=4, padding=0)
        self.drop1 = nn.Dropout()

        self.extra1 = nn.Conv2d(16, 64, (1, 1), stride=(1, 16))

        self.conv2 = nn.Conv2d(16, 32, (1, 64+1), padding=(0, 32))
        self.maxpool2 = nn.MaxPool2d((1, 4), stride=4, padding=0)
        self.conv3 = nn.Conv2d(32, 64, (1, 16+1), padding=(0, 8))
        self.maxpool3 = nn.MaxPool2d((1, 4), padding=0)
        self.drop3 = nn.Dropout()

        self.extra2 = nn.Conv2d(64, 256, (1, 1), stride=(1, 4))

        self.conv4 = nn.Conv2d(64, 128, (1, 4+1), padding=(0, 2))
        self.maxpool4 = nn.MaxPool2d((1, 2), stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, 256, (1, 2+1), padding=(0, 1))
        self.maxpool5 = nn.MaxPool2d((1, 2), stride=2, padding=0)
        self.drop5 = nn.Dropout()

        self.flatten = nn.Flatten(1, 3)

    def forward(self, x):
        out1 = F.relu(self.drop1(self.maxpool(self.conv1(x))))

        # residual block 1
        out2 = self.drop3(self.maxpool3(self.conv3(F.relu(self.maxpool2(self.conv2(out1))))))
        out3 = F.relu(self.extra1(out1)+out2)

        # residual block 2
        out4 = self.drop5(self.maxpool5(self.conv5(F.relu(self.maxpool4(self.conv4(out3))))))
        out5 = F.relu(self.extra2(out3)+out4)

        out = self.flatten(out5)
        return out
    
    def output_num(self):
        return 4096


class ResNetBackbone_Attention(nn.Module):
    def __init__(self):
        super(ResNetBackbone_Attention, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 128 + 1), padding=(0, 64))
        self.maxpool = nn.MaxPool2d((1, 4), stride=4, padding=0)
        self.drop1 = nn.Dropout()

        self.extra1 = nn.Conv2d(16, 64, (1, 1), stride=(1, 16))

        self.conv2 = nn.Conv2d(16, 32, (1, 64 + 1), padding=(0, 32))
        self.maxpool2 = nn.MaxPool2d((1, 4), stride=4, padding=0)
        self.conv3 = nn.Conv2d(32, 64, (1, 16 + 1), padding=(0, 8))
        self.maxpool3 = nn.MaxPool2d((1, 4), padding=0)
        self.drop3 = nn.Dropout()

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.extra2 = nn.Conv2d(64, 256, (1, 1), stride=(1, 4))

        self.conv4 = nn.Conv2d(64, 128, (1, 4 + 1), padding=(0, 2))
        self.maxpool4 = nn.MaxPool2d((1, 2), stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, 256, (1, 2 + 1), padding=(0, 1))
        self.maxpool5 = nn.MaxPool2d((1, 2), stride=2, padding=0)
        self.drop5 = nn.Dropout()

        self.flatten = nn.Flatten(1, 3)

    def forward(self, x):
        out1 = F.relu(self.drop1(self.maxpool(self.conv1(x))))

        # residual block 1
        out2 = self.drop3(self.maxpool3(self.conv3(F.relu(self.maxpool2(self.conv2(out1))))))
        out3 = F.relu(self.extra1(out1) + out2)

        out4 = self.ca(out3) * out3
        out5 = self.sa(out4) * out4

        # residual block 2
        out6 = self.drop5(self.maxpool5(self.conv5(F.relu(self.maxpool4(self.conv4(out5))))))
        out7 = F.relu(self.extra2(out5) + out6)

        out = self.flatten(out7)
        return out

    def output_num(self):
        return 16384


class ResNetBackbone_2D(nn.Module):
    def __init__(self):
        super(ResNetBackbone_2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (2, 128), padding=(1, 64))
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        out = self.maxpool(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        return out

    def output_num(self):
        return self._feature_dim

class AlexNetBackbone_2level(nn.Module):
    def __init__(self):
        super(AlexNetBackbone_2level, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 128), padding=(0, 64))
        self.maxpool1 = nn.MaxPool2d((1, 8))
        self.droup1 = nn.Dropout()

        self.conv2 = nn.Conv2d(16, 32, (1, 64), padding=(0, 32))
        self.maxpool2 = nn.MaxPool2d((1, 4))
        self.droup2 = nn.Dropout()

        self.flatten = nn.Flatten(1, 3)

    def forward(self, x):
        x = F.relu(self.droup1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.droup2(self.maxpool2(self.conv2(x))))
        out = self.flatten(x)
        return out

    def output_num(self):
        return 4096



# 以下为网络模块

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
