import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        net = []
        # 1:设定每次反卷积的输入和输出通道数
        #   卷积核尺寸固定为3，反卷积输出为“SAME”模式
        channels_in = [self.z_dim+self.num_classes, 512, 256, 128, 64]
        channels_out = [512, 256, 128, 64, 3]
        active = ["R", "R", "R", "R", "tanh"]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        for i in range(len(channels_in)):
            net.append(nn.ConvTranspose2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                          kernel_size=4, stride=stride[i], padding=padding[i], bias=False))
            if active[i] == "R":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.ReLU())
            elif active[i] == "tanh":
                net.append(nn.Tanh())

        self.generator = nn.Sequential(*net)

    def forward(self, x, label):
        data = torch.cat(tensors=(x, label), dim=1)
        out = self.generator(data)
        return out

if __name__ == '__main__':
    model = Generator(118, 10)
    x = torch.randn((118,))
    l = torch.randn((10,))
    out = model(x,l)
    print(out.shape)