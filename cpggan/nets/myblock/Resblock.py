import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.myblock.defconv import DCnv2


# 并联
# class ResBlock_g1(nn.Module):
#     def __init__(self, ch_in, ch_out, bn):
#
#         super(ResBlock_g1, self).__init__()
#
#         self.bn = bn
#
#         self.TConv = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1)
#         self.TConv_bn = nn.BatchNorm2d(ch_out)
#         self.TConv_relu = nn.ReLU()
#
#
#         self.Dconv_1 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=1, padding=1, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         self.Dconv_3 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=3, padding=3, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
#
#         self.Dconv_5 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=5, padding=5, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         # 换成普通3x3卷积 ?
#         self.Dconv_7 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=7, padding=7, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         self.conv_11 = nn.Conv2d(ch_out * 4, ch_out, kernel_size=1, bias=False)
#         self.conv_11_bn = nn.BatchNorm2d(ch_out)
#
#         self.conv_11_relu = nn.ReLU()
#
#
#
#     def forward(self, x):
#
#         x1 = self.TConv_relu(self.TConv_bn(self.TConv(x)))
#
#         x2_1 = self.Dconv_1(x1) + x1
#         x2_2 = self.Dconv_3(x1) + x1
#         x2_3 = self.Dconv_5(x1) + x1
#         x2_4 = self.Dconv_7(x1) + x1
#
#         x3 = torch.cat((x2_1,x2_2,x2_3,x2_4), dim=1)
#
#         if self.bn == True:
#             out = self.conv_11_relu(self.conv_11_bn(self.conv_11(x3)))
#
#         else:
#             out = torch.tanh(self.conv_11(x3))  # 生成器最后一层不用 bn，激活用tanh
#
#         return out


# -------------------------------------------------------------

# class ResBlock_g1(nn.Module):
#     def __init__(self, ch_in, ch_out, bn):
#
#         super(ResBlock_g1, self).__init__()
#
#         self.bn = bn
#
#         self.TConv = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1)
#         self.TConv_bn = nn.BatchNorm2d(ch_out)
#
#
#         self.relu = nn.ReLU()
#         self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
#
#
#
#         self.Dconv_1 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=1, padding=1, bias=False),
#             nn.BatchNorm2d(ch_out),
#             # nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         self.Dconv_3 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=3, padding=3, bias=False),
#             nn.BatchNorm2d(ch_out),
#             # nn.LeakyReLU(negative_slope=0.2),
#         )
#
#         self.Dconv_5 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=5, padding=5, bias=False),
#             nn.BatchNorm2d(ch_out),
#             # nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         self.Dconv_7 = nn.Sequential(
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=7, padding=7, bias=False),
#             nn.BatchNorm2d(ch_out),
#             # nn.LeakyReLU(negative_slope=0.2),
#         )
#
#
#         self.conv_33_1 = nn.Conv2d(ch_out * 2, ch_out, kernel_size=3,padding=1, bias=False)
#         self.conv_33_1_bn = nn.BatchNorm2d(ch_out)
#         self.conv_33_1_relu = nn.ReLU()
#
#         self.conv_33_2 = nn.Conv2d(ch_out * 2, ch_out, kernel_size=3, padding=1, bias=False)
#         self.conv_33_2_bn = nn.BatchNorm2d(ch_out)
#         self.conv_33_2_relu = nn.ReLU()
#
#         self.conv_11 = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, bias=False)
#         self.conv_11_bn = nn.BatchNorm2d(ch_out)
#         self.conv_11_relu = nn.ReLU()
#
#
#
#     def forward(self, x):
#
#         x1 = self.relu(self.TConv_bn(self.TConv(x)))
#
#         x2_1 = self.LeakyReLU(self.Dconv_1(x1) + x1)
#         x2_2 = self.LeakyReLU(self.Dconv_3(x1) + x1)
#         x2_3 = self.LeakyReLU(self.Dconv_5(x1) + x1)
#         x2_4 = self.LeakyReLU(self.Dconv_7(x1) + x1)
#
#         x3_1 = torch.cat((x2_1,x2_2), dim=1)
#         x3_2 = torch.cat((x2_3, x2_4), dim=1)
#
#         x4_1 = self.conv_33_1_relu(self.conv_33_1_bn(self.conv_33_1(x3_1)))
#         x4_2 = self.conv_33_2_relu(self.conv_33_2_bn(self.conv_33_2(x3_2)))
#
#         x5 = torch.cat((x4_1,x4_2), dim=1)
#
#
#         if self.bn == True:
#             out = self.conv_11_relu(self.conv_11_bn(self.conv_11(x5)))
#
#         else:
#             out = torch.tanh(self.conv_11(x5))  # 生成器最后一层不用 bn，激活用tanh
#
#         return out



# ---------------------------------------------

# class BasicConv(nn.Module):
#
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         if bn:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
#             self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             self.relu = nn.ReLU(inplace=True) if relu else None
#         else:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
#             self.bn = None
#             self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class BasicRFB(nn.Module):
#
#     def __init__(self, in_planes, out_planes, bn=True):  # scale=0.1   map_reduce=8
#         super(BasicRFB, self).__init__()
#
#         self.out_channels = out_planes
#         self.mybn = bn
#
#         inter_ch = in_planes // 2
#
#
#         self.TConv = nn.ConvTranspose2d(in_planes, inter_ch, 4, 2, 1)
#         self.bn1 = nn.BatchNorm2d(inter_ch)
#
#
#         self.branch0 = nn.Sequential(
#
#             BasicConv(inter_ch, inter_ch // 2, kernel_size=1, stride=1, bn=True, relu=True),
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=3, dilation=1, padding=1, bn=True, relu=False),
#         )
#
#         self.branch1 = nn.Sequential(
#
#             BasicConv(inter_ch, inter_ch // 2, kernel_size=1, stride=1, bn=True, relu=True),
#
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=(1, 3), padding=(0, 1), groups=inter_ch // 2, bn=True, relu=True),
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=(3, 1), padding=(1, 0), groups=inter_ch // 2, bn=True, relu=True),
#
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=3, dilation=3, padding=3, bn=True, relu=False),
#         )
#
#         self.branch2 = nn.Sequential(
#
#             BasicConv(inter_ch, inter_ch // 2, kernel_size=1, stride=1, bn=True, relu=True),
#
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=(3, 1), padding=(1, 0), groups=inter_ch // 2,bn=True, relu=True),
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=(1, 3), padding=(0, 1), groups=inter_ch // 2,bn=True, relu=True),
#
#             BasicConv(inter_ch // 2, inter_ch // 2, kernel_size=3, dilation=5, padding=5, bn=True, relu=False),
#         )
#
#         # self.branch3 = nn.Sequential(
#         #
#         #     nn.AdaptiveAvgPool2d((1, 1)),
#         #     BasicConv(in_planes, in_planes // 2, kernel_size=1, stride=1, bn=True, relu=False),
#         # )
#
#
#         self.ConvLinear = nn.Conv2d((inter_ch // 2) * 3, out_planes, kernel_size=1, stride=1)
#
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=False)
#
#
#     def forward(self, x):
#
#         y0 = self.relu(self.bn1(self.TConv(x)))
#         size = y0.shape[2:]
#
#         x0 = self.branch0(y0)
#         x1 = self.branch1(y0)
#         x2 = self.branch2(y0)
#
#         # x3 = self.branch3(y0)
#         # x3_1 = F.interpolate(x3, size=size, mode='bilinear',align_corners=True)
#
#
#         x4 = self.relu(torch.cat((x0, x1, x2), dim=1))
#
#
#         if self.mybn is True:
#             out = self.relu(self.bn2(self.ConvLinear(x4)))
#
#         else:
#             out = torch.tanh(self.ConvLinear(x4))
#
#
#         return out

# ------------------------------------------------


class DSConv(nn.Module):

    def __init__(self, in_channel, out_channel, ksize=3,padding=1,bais=True):
        super(DSConv, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=in_channel,
                                       groups=in_channel,
                                       kernel_size=ksize,
                                       padding=padding,
                                       bias=bais)

        self.bn = nn.BatchNorm2d(num_features=in_channel)
        self.relu = nn.ReLU()

        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       bias=bais)

    def forward(self, x):

        x1 = self.relu(self.bn(self.depthwiseConv(x)))
        out = self.pointwiseConv(x1)

        return out



#
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x




class ConvBlock_A(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock_A, self).__init__()

        self.branch_1 = nn.Sequential(

            # nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=1 ,padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,dilation=3,padding=3, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
        )

    def forward(self, x):

        out = self.branch_1(x)
        return out




class ConvBlock_B(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock_B, self).__init__()

        self.branch_1 = nn.Sequential(

            DSConv(in_ch, out_ch),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

        )

    def forward(self, x):

        out = self.branch_1(x)
        return out



class ConvBlock_C(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock_C, self).__init__()

        self.branch_1 = nn.Sequential(

            CoordConv(in_ch, out_ch),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
        )

    def forward(self, x):

        out = self.branch_1(x)
        return out



# class ResBlock_g1(nn.Module):
#     def __init__(self, ch_in, ch_out, bn):
#
#         super(ResBlock_g1, self).__init__()
#
#         self.mybn = bn
#
#         self.lay_1 = nn.Sequential(
#
#             nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1),
#             nn.BatchNorm2d(num_features=ch_out),
#             nn.ReLU(),
#         )
#
#
#         self.block_1 = ConvBlock_A(ch_out,ch_out)
#         self.block_2 = ConvBlock_B(ch_out, ch_out)
#         self.block_3 = ConvBlock_C(ch_out, ch_out)
#
#         self.Conv_11 = nn.Conv2d(ch_out * 3, ch_out, kernel_size=1, stride=1, bias=False)
#
#         self.bn = nn.BatchNorm2d(ch_out)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#
#         x0 = self.lay_1(x)
#
#         x1 = self.block_1(x0)
#         x2 = self.block_2(x0)
#         x3 = self.block_3(x0)
#
#         x4 = torch.cat((x3, x1, x2), dim=1)
#
#         if self.mybn == True:
#             out = self.relu(self.bn(self.Conv_11(x4)))
#         else:
#             out = torch.tanh(self.Conv_11(x4))
#
#         return out



class ResBlock_g1(nn.Module):
    def __init__(self, ch_in, ch_out, bn):

        super(ResBlock_g1, self).__init__()

        self.bn = bn
        self.alph = 0.3  # 定义的残差因子


        self.TConv = nn.ConvTranspose2d(ch_in, ch_in, 4, 2, 1)
        self.TConv_bn = nn.BatchNorm2d(ch_in)
        self.TConv_relu = nn.ReLU()



        self.Dconv_0 = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )


        self.Dconv_1 = nn.Sequential(

            # nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            # #nn.BatchNorm2d(ch_out),
            # nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=1,stride=1,groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            #nn.ReLU(),
        )

        self.Dconv_3 = nn.Sequential(

            # nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            # #nn.BatchNorm2d(ch_out),
            # nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=3, padding=3, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            #nn.ReLU(),
        )

        self.Dconv_5 = nn.Sequential(

            # nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            # #nn.BatchNorm2d(ch_out),
            # nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=3, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            #nn.ReLU(),
        )


        self.DFconv = nn.Sequential(

            # nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            # # nn.BatchNorm2d(ch_out),
            # nn.ReLU(),


            DCnv2(ch_out, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

        )



        self.conv_11 = nn.Conv2d(ch_out * 4, ch_out, kernel_size=1, bias=False)
        self.conv_11_bn = nn.BatchNorm2d(ch_out)

        self.conv_11_relu = nn.ReLU()


    def forward(self, x):

        x0 = self.TConv_relu(self.TConv_bn(self.TConv(x)))
        x1 = self.Dconv_0(x0)


        x2_1 = self.Dconv_1(x1) + x1
        x2_2 = self.Dconv_3(x1) + x1
        x2_3 = self.Dconv_5(x1) + x1
        x2_4 = self.DFconv(x1) + x1


        x3 = torch.cat((x2_1,x2_2,x2_3,x2_4), dim=1)


        if self.bn == True:
            out = self.conv_11_relu(self.conv_11_bn(self.conv_11(x3)))
        else:
            out = torch.tanh(self.conv_11(x3))

        return out