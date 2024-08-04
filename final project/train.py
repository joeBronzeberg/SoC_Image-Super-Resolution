import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from dataset import SRDataset
from model import FSRCNN
from utils import PSNR, AverageMeter, save_evaluation_image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()


model = FSRCNN(scale_factor=cfg.scale_factor, d=cfg.d, c=cfg.c, s=cfg.s, m=cfg.m)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), cfg.learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_learning_rate, verbose=False
)

dataloader = DataLoader(
    SRDataset(f"./data/{cfg.dataset_name}", hr_shape=cfg.hr_shape, scale_factor=cfg.scale_factor),
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.n_cpu,
)

val_dataloader = DataLoader(
    SRDataset(f"./data/{cfg.val_dataset_name}", hr_shape=cfg.hr_shape, scale_factor=cfg.scale_factor),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.n_cpu,
)

test_dataloader = DataLoader(
    SRDataset(f"./data/{cfg.test_dataset_name}", hr_shape=cfg.hr_shape, scale_factor=cfg.scale_factor),
    batch_size=cfg.test_batch_size,
    shuffle=False,
    num_workers=1,
)

for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = AverageMeter()

    for lr_imgs, hr_imgs in dataloader:

        optimizer.zero_grad()

        sr_imgs = model(lr_imgs)

        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item(),len(lr_imgs))

    val_loss = AverageMeter()
    model.eval()

    for lr_imgs, hr_imgs in val_dataloader:

        with torch.no_grad():
            sr_imgs = model(lr_imgs).clamp(0.0, 1.0)
        
        loss = criterion(sr_imgs, hr_imgs)
          
        val_loss.update(loss.item(),len(lr_imgs))

    scheduler.step(val_loss.avg)

    print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Train Loss: {epoch_loss.avg:.6f}, Val Loss: {val_loss.avg:.6f}")

    if (epoch + 1) % 10 == 0:

        model.eval()
        for lr_imgs, hr_imgs in test_dataloader:
            with torch.no_grad():
                sr_imgs = model(lr_imgs).clamp(0.0, 1.0)
            lr_tensor = lr_imgs[0]
            hr_tensor = hr_imgs[0]
            sr_tensor = sr_imgs[0]
            save_evaluation_image(epoch,lr_tensor, hr_tensor, sr_tensor, f"training/{cfg.model_name}")
        
        torch.save(model.state_dict(), f"./saved_models/{cfg.model_name}/fsrcnnx4_epoch_{epoch+1}.pth")

