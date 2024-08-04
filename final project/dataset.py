import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, root_dir, hr_shape, scale_factor):
        self.root_dir = root_dir
        self.hr_shape = hr_shape
        self.scale_factor = scale_factor
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        lr_size = (hr_shape[0] // scale_factor, hr_shape[1] // scale_factor)

        self.transform = transforms.Compose([
            transforms.RandomCrop(min(hr_shape)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_shape),
            transforms.ToTensor(),
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path)

        img = self.transform(img)

        hr_img = self.hr_transform(img)

        lr_img = self.lr_transform(img)

        return lr_img, hr_img

