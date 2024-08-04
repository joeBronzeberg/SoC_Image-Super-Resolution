import torch
import numpy as np
import os
from PIL import Image


def PSNR(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tensor_to_pil(image_tensor):

    image_numpy = image_tensor.detach().numpy()
    return Image.fromarray(np.uint8(image_numpy * 255), mode='L')

def save_evaluation_image(epoch, lr_tensor, hr_tensor, sr_tensor, save_dir):
    
    os.makedirs(save_dir, exist_ok=True)
    
    lr_img = tensor_to_pil(lr_tensor.squeeze(0))
    hr_img = tensor_to_pil(hr_tensor.squeeze(0))
    sr_img = tensor_to_pil(sr_tensor.squeeze(0))

    lr_img_resized = lr_img.resize(hr_img.size, Image.NEAREST)
    
    width, height = hr_img.size
    total_width = width * 3
    
    new_image = Image.new('L', (total_width, height))
    
    new_image.paste(lr_img_resized, (0, 0))
    new_image.paste(hr_img, (width, 0))
    new_image.paste(sr_img, (width * 2, 0))
    
    save_path = os.path.join(save_dir, f'test_epoch_{epoch+1}.png')
    new_image.save(save_path)
