import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from PIL import Image
from torch import nn
from utils.utils import postprocess_output, show_config
from nets.cpggan import Discriminator, Generator



class DCGAN(object):
    _defaults = {
        "model_path": r'',
        "channel": 64,
        "input_shape": [256, 256],
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.generate()

        show_config(**self._defaults)

    def generate(self):

        self.net = Generator(self.channel, self.input_shape).eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def generate_5x5_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((5 * 5, 100))
            if self.cuda:
                randn_in = randn_in.cuda()

            test_images = self.net(randn_in)

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5 * 5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(np.uint8(postprocess_output(test_images[k].cpu().data.numpy().transpose(1, 2, 0))))

            label = 'predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)


    def generate_1x1_image(self, save_path):
        with torch.no_grad():

            start_time = time.time()

            randn_in = torch.randn((1, 100))
            if self.cuda:
                randn_in = randn_in.cuda()

            test_images = self.net(randn_in)
            test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))


            Image.fromarray(np.uint8(test_images)).save(save_path)

            end_time = time.time()
            elapsed_time_sec = end_time - start_time
            return elapsed_time_sec


if __name__ == "__main__":

    dcgan = DCGAN()

    save_path_1x1 = r""

    num = 5000
    if num < 2:
        raise ValueError("11")

    generation_times = []

    for i in range(num):

        img_path = save_path_1x1 + f'gan_{i}.png'

        elapsed_time = dcgan.generate_1x1_image(img_path)
        generation_times.append(elapsed_time)

        print(f'生成第 {i - 300}/{num} 张图片，耗时: {elapsed_time:.4f} 秒')

    valid_times = generation_times[1:]
    avg_time = sum(valid_times) / len(valid_times)

    print(f'\n共生成 {num} 张图片，剔除第一张后的统计:')
    print(f'有效样本数: {len(valid_times)} 张')
    print(f'单张图片生成平均时间: {avg_time:.4f} 秒')
    print(f'生成时间范围: {min(valid_times):.4f} - {max(valid_times):.4f} 秒')