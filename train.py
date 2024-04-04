import os
import torch
import yaml
import torch, gc

from utils import network_parameters, losses
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
import random
from transform.data_RGB import get_training_data, get_validation_data2
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import utils.losses
from model.CF_UFormer import CF_UFormer
import argparse
import Myloss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='Hyper-parameters for CF-UFormer')
parser.add_argument('-yml_path', default="./configs/LSRW/train/training_LSRW.yaml", type=str)
args = parser.parse_args()

# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Load yaml configuration file
yaml_file = args.yml_path

with open(yaml_file, 'r') as config:
    opt = yaml.safe_load(config)
print("load training yaml file: %s" % yaml_file)

Train = opt['TRAINING']
OPT = opt['OPTIM']

# Build Model
print('==> Build the model')
model_CF_UFormer = CF_UFormer(inp_channels=3, out_channels=3, dim=16, num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
                          heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                          attention=True, skip=False)
para_number = network_parameters(model_CF_UFormer)
model_CF_UFormer.cuda()

# Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# GPU setting
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_CF_UFormer = nn.DataParallel(model_CF_UFormer, device_ids=device_ids)

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_CF_UFormer.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Resuming
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_CF_UFormer, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

# Loss
L_SL1 = nn.SmoothL1Loss()
L_col = Myloss.L_color()
L_spa = Myloss.L_spa()
L_exp = Myloss.L_exp(16)
L_ssim = Myloss.SSIM()
L_per = Myloss.VGGLoss(dev)

# DataLoader
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=8, drop_last=False)
val_dataset = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {para_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_PSNR = 0
best_SSIM = 0
best_epoch_PSNR = 0
best_epoch_SSIM = 0
total_start_time = time.time()

# logging
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_CF_UFormer.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # Forward propagation
        for param in model_CF_UFormer.parameters():
            param.grad = None
        E = 0.6
        target = data[0].cuda()
        input_img = data[1].cuda()
        enhanced_img = model_CF_UFormer(input_img)

        loss_SL1 = L_SL1(enhanced_img, target)
        # print(loss_SL1)
        loss_ssim = 1 - L_ssim(enhanced_img, target)
        # print(loss_ssim)
        loss_spa = torch.mean(L_spa(input_img, enhanced_img))
        # print(loss_spa)
        loss_col = 5 * torch.mean(L_col(enhanced_img))
        loss_exp = 10 * torch.mean(L_exp(enhanced_img, E))
        loss_per = L_per(enhanced_img, target)
        # print(loss_per)
        # loss = loss_SL1 + 0.1*loss_ssim + 0.1*loss_spa + 0.1*loss_per
        loss = loss_SL1 + 0.1*loss_ssim + 0.1*loss_spa + 0.1*loss_per

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_CF_UFormer.eval()
        PSNR_val_rgb = []
        SSIM_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_img = data_val[1].cuda()
            h, w = target.shape[2], target.shape[3]
            with torch.no_grad():
                enhanced_img = model_CF_UFormer(input_img)
                enhanced_img = enhanced_img[:, :, :h, :w]

            for res, tar in zip(enhanced_img, target):
                PSNR_val_rgb.append(utils.torchPSNR(res, tar))
                SSIM_val_rgb.append(utils.torchSSIM(enhanced_img, target))

        PSNR_val_rgb = torch.stack(PSNR_val_rgb).mean().item()
        SSIM_val_rgb = torch.stack(SSIM_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if PSNR_val_rgb > best_PSNR:
            best_PSNR = PSNR_val_rgb
            best_epoch_PSNR = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_CF_UFormer.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, PSNR_val_rgb, best_epoch_PSNR, best_PSNR))

        # Save the best SSIM model of validation
        if SSIM_val_rgb > best_SSIM:
            best_SSIM = SSIM_val_rgb
            best_epoch_SSIM = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_CF_UFormer.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, SSIM_val_rgb, best_epoch_SSIM, best_SSIM))

        """
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', PSNR_val_rgb, epoch)
        writer.add_scalar('val/SSIM', SSIM_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_CF_UFormer.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
