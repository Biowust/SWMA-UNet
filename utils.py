import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
import logging
from matplotlib import pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as nF
import cv2
from torchvision import models, transforms
from torch.optim.optimizer import Optimizer
import weakref
from functools import wraps
import math
from sklearn.metrics import roc_auc_score,jaccard_score
from GradCAM import GradCAM

import copy
from PIL import Image
from GradCAM import show_cam_on_image
from skimage import morphology

class DiceLoss(nn.Module):
    def __init__(self, n_classes=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-6  # 提高平滑项
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = torch.ones(self.n_classes, dtype=inputs.dtype, device=inputs.device)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes  # 对类的总权重进行归一化处理可能会更稳健

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    grad_cam = GradCAM(net, target_layers=[net.swma_unet.output])  # Adjust target layer for the model
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)  # Resize for patch compatibility
        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            pred = out
        prediction[ind] = pred
        mode_class = torch.mode(torch.flatten(torch.tensor(out))).values.item()
        # Generate CAM for each slice
        grayscale_cam = grad_cam(input_tensor=input_tensor, target_category=mode_class)
        grayscale_cam = grayscale_cam[0, :]
        slice_save_path = os.path.join(test_save_path, f"{case}_slice_{ind}_grad_cam_overlay.jpg")
        save_cam_heatmap(slice_img, grayscale_cam, slice_save_path)

    # Calculate metrics per class
    metric_list = [calculate_metric_percase(prediction == i, label == i) for i in range(1, classes)]

    # Save prediction and original images in ITK format
    if test_save_path:
        os.makedirs(test_save_path, exist_ok=True)
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f'{case}_pred.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f'{case}_img.nii.gz'))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f'{case}_gt.nii.gz'))

    return metric_list

import matplotlib.pyplot as plt
from matplotlib import cm

def apply_custom_colormap(grayscale_cam, img, save_path):
    """
    应用自定义红蓝颜色映射，叠加注意力热图。
    """
    # 确保 grayscale_cam 范围在 [0, 1]
    grayscale_cam = np.clip(grayscale_cam, 0, 1)
    
    # 使用 'hot' 或其他自定义映射
    colormap = cm.get_cmap('RdBu')  # 或 'seismic' 等映射
    heatmap = colormap(grayscale_cam)[:, :, :3]  # 获取 RGB 通道

    # 确保原图在 [0, 1]
    if np.max(img) > 1.0:
        img = img / 255.0
    # 转换为 uint8 类型以匹配 OpenCV 需求
    heatmap = (heatmap * 255).astype(np.uint8)
    img = (img * 255).astype(np.uint8)
    # 将热图与原图融合，按比例叠加
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 保存叠加后的结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_cam_heatmap(image, grayscale_cam, save_path, overlay=True,cmap='coolwarm'):
    # 确保输入图像是 numpy 数组
    if isinstance(image, Image.Image):  # 如果是 PIL 图像
        image = np.array(image)

    # 确保图像范围在 [0, 1]
    if np.max(image) > 1.0:
        image = image / 255.0

    # 确保图像是 RGB 三通道
    if len(image.shape) == 2:  # 如果是灰度图像
        image = np.stack([image] * 3, axis=-1)

    if overlay:
        # 应用自定义颜色映射
        apply_custom_colormap(grayscale_cam, image, save_path)
    else:
        # 如果仅保存热图
        plt.imshow(grayscale_cam, cmap=cmap)
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss_()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = bceloss*self.wb + diceloss*self.wd
        return loss

class DiceLoss_(nn.Module):
    def __init__(self):
        super(DiceLoss_, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)
        
        intersection = pred_ * target_
        
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)

        return self.bceloss(pred_, target_)

def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss

class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape == truth.shape)
        
        # 计算正负样本的权重
        pos_weight = (truth > 0.5).float().sum() + 1e-12
        neg_weight = (truth < 0.5).float().sum() + 1e-12
        pos_weight_tensor = torch.tensor(self.weights[0] / pos_weight).to(logit.device)
        neg_weight_tensor = torch.tensor(self.weights[1] / neg_weight).to(logit.device)
        
        # 使用 BCEWithLogitsLoss 并通过 pos_weight 参数来实现加权
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor / neg_weight_tensor)
        loss = criterion(logit, truth)

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]

        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)

        loss = dice.mean()
        return loss

class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch


        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method

            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
