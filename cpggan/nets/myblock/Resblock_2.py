import torch.nn as nn

class ResBlock_d_a(nn.Module):
    def __init__(self, ch_in, ch_out, bn=True):

        super(ResBlock_d_a, self).__init__()

        self.bn = bn

        self.branch_0 = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, 4, 2, 1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.2),

        )

        self.branch_1 = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, 4, 2, 1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.2),

        )


    def forward(self, x):

        if self.bn == False:
            out = self.branch_0(x)
        else:
            out = self.branch_1(x)

        return out



class ResBlock_d_b(nn.Module):
    def __init__(self, ch_in, ch_out):

        super(ResBlock_d_b, self).__init__()

        self.branch_0 = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, 4, 2, 1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.2),

        )

    def forward(self, x):

        out = self.branch_0(x)

        return out