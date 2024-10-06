import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from medpy.metric.binary import dc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from torchinfo import summary
from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
from datasets.dataset_ISIC import isic_loader
import logging
from utils import get_logger,save_imgs,BceDiceLoss
from sklearn.metrics import confusion_matrix


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95

# def inference(model, testloader, args, test_save_path=None):
#     model.eval()
#     metric_list = 0.0

#     # Grad-CAM实例，指定最后的卷积层
#     grad_cam = GradCAM(model, target_layer="swin_unet.output")  # 替换成你的模型的最后一个卷积层名称

#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         image = image.cuda()

#         # 正常推理过程
#         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)

#         # 生成并保存Grad-CAM热图
#         class_idx = label.argmax().item()  # 获取目标类别的索引
#         cam = grad_cam.generate_cam(image, class_idx)
        
#         cam_save_path = os.path.join(test_save_path, f"{case_name}_grad_cam.jpg")
#         save_cam_image(cam, sampled_batch['image_path'][0], cam_save_path)
#         logging.info(f'Saved Grad-CAM for {case_name} at {cam_save_path}')
        
#         logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

#     metric_list = metric_list / len(testloader.dataset)

#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]

#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

#     return performance, mean_hd95

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    test_save_path = os.path.join(snapshot_path, 'test')
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    print(summary(model, input_size=(batch_size, 3, 224, 224)))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # Load checkpoint if it exists
    start_epoch = 0
    best_performance = 0.0
    Max_dice = 0.75  # Initialize Max_dice here as a fallback

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    if args.resume:
        checkpoint_path = os.path.join(snapshot_path, args.resume)
        # print(f"Loading checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_performance = checkpoint['best_performance']
            Max_dice = checkpoint['Max_dice']
            logging.info(f"Resuming training from epoch {start_epoch} with best_performance {best_performance}")
            iter_num = start_epoch * len(trainloader)  # Adjust iter_num based on start_epoch
        else:
            logging.info(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            iter_num = 0  # If starting from scratch, iter_num starts from 0
    else:
        iter_num = 0  # If not resuming, iter_num starts from 0

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)  # Start iterator from start_epoch

    dice_ = []
    hd95_ = []
    loss_history = []
    fig, ax = plt.subplots()

    for epoch_num in iterator:
        running_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1  # Continue iter_num from where it left off
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            running_loss += loss.item()

            try:
                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except: pass

        # mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                    
        if epoch_num % 10 == 0:
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(snapshot_path, f'latest.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_performance': best_performance,
                'Max_dice': Max_dice,
            }, checkpoint_path)

        # Eval and save model as needed
        if (epoch_num + 1) % 50 == 0 and (epoch_num + 1) > 50:
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            if mean_dice > Max_dice:
                filename = f'epoch_{epoch_num}_{mean_dice}.pth'
                save_mode_path = os.path.join(snapshot_path, filename)
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"Saved model to {save_mode_path}")
                Max_dice = mean_dice

            

        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        logging.info(f'Epoch [{epoch_num+1}/{max_epoch}], Loss: {avg_loss:.4f}')

        plt.figure()
        ax = plt.gca()
        ax.plot(loss_history)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join(snapshot_path, f"loss.png"))
        plt.close()

    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()

    # generate_attention_maps(model, testloader, snapshot_path)
    return "Training Finished!"

def trainer_ACDC(args, model, snapshot_path):
    # Set up logging
    log_file = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Hyperparameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Datasets and Data Loaders
    train_transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", transform=train_transform)
    db_val = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    db_test = ACDC_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    # Model setup for multiple GPUs
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Load checkpoint if it exists
    start_epoch = 0
    best_performance = 0.0
    Best_dcs = 0.90  # Initialize Best_dcs here as a fallback

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    if args.resume:
        checkpoint_path = os.path.join(snapshot_path, args.resume)
        # print(f"Loading checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_performance = checkpoint['best_performance']
            Best_dcs = checkpoint['Best_dcs']
            logging.info(f"Resuming training from epoch {start_epoch} with best_performance {best_performance}")
            iter_num = start_epoch * len(trainloader)  # Adjust iter_num based on start_epoch
        else:
            logging.info(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            iter_num = 0  # If starting from scratch, iter_num starts from 0
    else:
        iter_num = 0  # If not resuming, iter_num starts from 0

    model.train()
    
    # Losses and Optimizer
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    # Lists to store evaluation metrics
    dice_ = []
    hd95_ = []
    loss_history = []

    for epoch_num in iterator:
        running_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs = model(image_batch)
            
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Adjust learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            # Logging
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            running_loss += loss.item()
            
            # Image logging
            if iter_num % 20 == 0:
                try:
                    image = (image_batch[1, 0:1, :, :] - image_batch.min()) / (image_batch.max() - image_batch.min())
                    writer.add_image('train/Image', image, iter_num)
                    writer.add_image('train/Prediction', torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)[1, ...] * 50, iter_num)
                    writer.add_image('train/GroundTruth', label_batch[1, ...].unsqueeze(0) * 50, iter_num)
                except Exception as e:
                    logging.warning(f"Image logging failed: {e}")
        
        # Epoch Loss Logging
        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        logging.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch_num+1, max_epoch, avg_loss))
        
        # Plot training loss
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(snapshot_path, "loss.png"))
        plt.close()

        # Save checkpoint after every epoch
        checkpoint_path = os.path.join(snapshot_path, 'latest.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_performance': best_performance,
            'Best_dcs': Best_dcs,
        }, checkpoint_path)

        # Periodic Evaluation and Model Saving
        if (epoch_num + 1) % args.eval_interval == 0 and (epoch_num + 1) > 99:
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=os.path.join(snapshot_path, 'test'))
            if mean_dice > Best_dcs:
                filename = f'epoch_{epoch_num}_{mean_dice}.pth'
                save_mode_path = os.path.join(snapshot_path, filename)
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"Saved model to {save_mode_path}")
                Best_dcs = mean_dice
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        # if epoch_num >= max_epoch - 1:
        #     # Final Model Save
        #     model_save_path = os.path.join(snapshot_path, f'{args.model_name}_epoch_{epoch_num}.pth')
        #     torch.save(model.state_dict(), model_save_path)
        #     logging.info(f"Final model saved at {model_save_path}")
            
        #     if (epoch_num + 1) % args.eval_interval != 0:
        #         mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=os.path.join(snapshot_path, 'test'))
        #         dice_.append(mean_dice)
        #         hd95_.append(mean_hd95)
        #         model.train()
        #     break

    # Plot Dice and HD95 results
    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    
    return "Training Finished!"

def trainer_ISIC(args, model, snapshot_path):
    # Set up logging
    log_file = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = get_logger('train', os.path.join(args.output_dir, 'log'))
    logger.info(str(args))

    # Hyperparameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Datasets and Data Loaders
    transform = transforms.Compose([
    transforms.Resize((224, 224))  # Resize the image to 224x224
    ])
    train_dataset = isic_loader(path_Data=args.root_path, train=True, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=8)
    val_dataset = isic_loader(path_Data=args.root_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=8, drop_last=True)
    test_dataset = isic_loader(path_Data=args.volume_path, train=False, Test=True, transform=transform)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=8, drop_last=True)

    # Model setup for multiple GPUs
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint if it exists
    start_epoch = 0
    best_f1_or_dsc = 0.90  # Default fallback
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)

    if args.resume:
        checkpoint_path = os.path.join(snapshot_path, args.resume)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1_or_dsc = checkpoint['best_f1_or_dsc']
            logger.info(f"Resuming training from epoch {start_epoch}")
            iter_num = start_epoch * len(trainloader)
        else:
            logger.info(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            iter_num = 0
    else:
        iter_num = 0

    model.train()

    # Losses and Optimizer
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)

    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    # Lists to store evaluation metrics
    miou_ = []
    f1_or_dsc_ = []
    loss_history = []
    
    for epoch_num in iterator:
        running_loss = 0.0
        for iter, data in enumerate(trainloader):
            bce_dice_loss = BceDiceLoss(0.3,0.7)
            optimizer.zero_grad()
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            outputs = torch.sigmoid(model(images))
            
            loss = bce_dice_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            # Logging
            iter_num += 1
            writer.add_scalar('info/lr', now_lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            running_loss += loss.item()

        # Epoch Loss Logging
        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch_num+1, max_epoch, avg_loss))

        # Plot training loss
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(snapshot_path, "loss.png"))
        plt.close()

        # Save checkpoint after every epoch
        checkpoint_path = os.path.join(snapshot_path, 'latest.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1_or_dsc': best_f1_or_dsc,
        }, checkpoint_path)

        # Periodic Evaluation and Model Saving
        if (epoch_num + 1) % args.eval_interval == 0 and (epoch_num + 1) > 99:
            loss, miou, f1_or_dsc = test_isic(testloader, model, logger,args)
            if f1_or_dsc > best_f1_or_dsc:
                filename = f'epoch_{epoch_num}_{f1_or_dsc:.4f}.pth'
                save_mode_path = os.path.join(snapshot_path, filename)
                torch.save(model.state_dict(), save_mode_path)
                logger.info(f"Saved model to {save_mode_path}")
                best_f1_or_dsc = f1_or_dsc
            f1_or_dsc_.append(best_f1_or_dsc)
            miou_.append(miou)
            model.train()

    plot_result(miou_, f1_or_dsc_, snapshot_path, args)
    writer.close()

    return "Training Finished!"

    

def test_isic(test_loader,
                    model,
                    logger,
                    args,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if type(data) is list:
                img, msk = data
            elif type(data) is dict:
                img, msk = data['image'], data['label']
            else:
                raise ValueError('data type is not list or dict')
            bce_dice_loss = BceDiceLoss()
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = torch.sigmoid(model(img))
            loss = bce_dice_loss(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            save_imgs(img, msk, out, i, args.output_dir + 'outputs/', args.dataset, 0.5, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=0.5, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list),miou,f1_or_dsc



def get_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print("@@@@",name)
    #         last_conv = name
    # return last_conv