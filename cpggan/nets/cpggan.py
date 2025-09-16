import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Generator(nn.Module):
    def __init__(self, d = 128, input_shape = [256, 256]):
        super(Generator, self).__init__()
        s_h, s_w    = input_shape[0], input_shape[1]
        s_h2, s_w2  = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4  = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8  = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        self.s_h16, self.s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


        self.linear = nn.Linear(100, self.s_h16 * self.s_w16 * d * 8)
        self.linear_bn = nn.BatchNorm2d(d * 8)


        self.ResBlock_g1 = ResBlock_g1(d * 8, d * 4, bn=True)

        self.EMA_1 = EMA(d * 4)

        self.ResBlock_g2 = ResBlock_g1(d * 4, d * 2, bn=True)

        self.EMA_2 = EMA(d * 2)

        self.ResBlock_g3 = ResBlock_g1(d * 2, d, bn=True)

        self.EMA_3 = EMA(d)

        self.ResBlock_g4 = ResBlock_g1(d, 3, bn=False)

        self.relu = nn.ReLU()
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        input_x = x

        bs, _ = x.size()
        x_01 = self.linear(x)
        x_02 = x_01.view([bs, -1, self.s_h16, self.s_w16])

        x1 = self.relu(self.linear_bn(x_02))


        x2 = self.ResBlock_g1(x1)

        x3 = self.EMA_1(x2) + x2

        x4 = self.ResBlock_g2(x3)

        x5 = self.EMA_2(x4) + x4

        x6 = self.ResBlock_g3(x5)

        x7 = self.EMA_3(x6) + x6

        out = self.ResBlock_g4(x7)

        return out


class Discriminator(nn.Module):
    def __init__(self, d = 128, input_shape = [256, 256]):
        super(Discriminator, self).__init__()

        s_h, s_w    = input_shape[0], input_shape[1]
        s_h2, s_w2  = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4  = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8  = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        self.s_h16, self.s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


        self.ResBlock_d1 = ResBlock_d_a(3, d, bn=False)

        self.ResBlock_d2 = ResBlock_d_b(d, d * 2)
        self.ResBlock_d3 = ResBlock_d_b(d * 2, d * 4)

        self.EMA = EMA(d * 4)

        self.ResBlock_d4 = ResBlock_d_a(d * 4, d * 8, bn=True)


        # 4,4,1024 -> 1,1,1
        self.linear     = nn.Linear(self.s_h16 * self.s_w16 * d * 8, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.weight_init()



    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):

        bs, _, _, _ = x.size()

        x1 = self.ResBlock_d1(x)
        x2 = self.ResBlock_d2(x1)
        x3 = self.ResBlock_d3(x2)

        x4 = self.EMA(x3) + x3

        x5 = self.ResBlock_d4(x4)

        x6 = x5.view([bs, -1])
        x7 = self.linear(x6)

        out = x7.squeeze()

        return out


if __name__ == "__main__":
    from torchsummary import summary
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = Generator(128).to(device)
    summary(model, input_size=(100,))

    model   = Discriminator(128).to(device)
    summary(model, input_size=(3, 64, 64))
