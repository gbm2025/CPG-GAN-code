import torch
from thop import clever_format, profile
from nets.cpggan import Generator,Discriminator  # CPG-GAN



### 1、对生成器-------------------------------------------------
# 创建生成器实例
model = Generator(d=128, input_shape=[256, 256])

# 生成输入张量
input_tensor = torch.randn(1, 100)

# 计算参数量和FLOPs
flops, params = profile(model, inputs=(input_tensor,))

print('FLOPs: ' + str(flops/1000**3) + 'G')  # 复杂度
print('Params: ' + str(params/1000**2) + 'M')  # 参数量



### 2、对判别器-------------------------------------------------
# 创建判别器实例
# discriminator = discriminator(d=128, input_shape=[256, 256])
#
# # 生成符合判别器输入的张量
# # 判别器输入是真实/生成图像，而非生成器的 latent vector
# input_tensor = torch.randn(1, 3, 256, 256)
#
# # 用thop计算FLOPs（浮点运算次数）和参数量
# flops, params = profile(discriminator, inputs=(input_tensor,))
#
# print('FLOPs: ' + str(flops/1000**3) + 'G')  # 复杂度
# print('Params: ' + str(params/1000**2) + 'M')  # 参数量