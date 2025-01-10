# -*- coding: utf-8 -*-
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from datasets.dataset_Monu import RandomGenerator,ValGenerator,ImageToImage2D,train_one_epoch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE
import optuna
from tqdm import tqdm

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    dice = state['max_dice']  # floatq
    

    filename = f'epoch_{epoch}_{dice}.pth'
    save_mode_path = os.path.join(save_path, filename)
    torch.save(state, save_mode_path)



##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
def trainer_MoNu(args, model, snapshot_path):
    logger = logger_config(log_path=snapshot_path+ "/log.txt")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # Load train and val data
    train_tf= transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    val_tf = ValGenerator(output_size=[args.img_size, args.img_size])
    train_dataset = ImageToImage2D(args.root_path, train_tf,image_size=args.img_size)
    val_dataset = ImageToImage2D(args.test_path, val_tf,image_size=args.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=1,
                            pin_memory=True)

    lr = args.base_lr

    model = model.cuda()
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])

    criterion = WeightedDiceBCE(dice_weight=args.dice_weight,BCE_weight=1-args.dice_weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    max_iou = 0.0
    max_dice = 0.0
    best_epoch = 1
    for epoch in tqdm(range(args.max_epochs), ncols=70):  # loop over the dataset multiple times
        model.train(True)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, None, logger,args)
        with torch.no_grad():
            model.eval()
            val_loss, val_dice, val_iou = train_one_epoch(val_loader, model, criterion,
                                            optimizer, epoch, lr_scheduler,logger,args)
            # trial.report(val_dice, epoch)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch+1 > 1:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                logger.info('\t mean iou: {:.4f}'.format(val_iou))
                max_dice = val_dice
                max_iou = val_iou
                best_epoch = epoch + 1
                if max_dice > 0.804:
                    save_checkpoint({'epoch': epoch,
                                    'best_model': True,
                                    'model_state_dict': model.state_dict(),
                                    'max_dice':max_dice,
                                    'val_loss': val_loss,
                                    'max_iou':max_iou,
                                    'optimizer_state_dict': optimizer.state_dict()}, snapshot_path)

        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} and {:.4f} in epoch {}'.format(val_dice,max_dice, max_iou, best_epoch))
    

    return max_dice, max_iou    

