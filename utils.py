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
import torch.nn.functional as F
import cv2
from torchvision import models, transforms


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
    grad_cam = GradCAM(net, target_layer="swma_unet.output")  # 根据模型调整最后一层的名字

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            pred = out
        prediction[ind] = pred

        # 生成热图
        # mode_class = torch.mode(torch.flatten(out)).values.item()  # 使用 mode 计算主要类别
        mode_class = torch.mode(torch.flatten(torch.tensor(out))).values.item()  # 使用 mode 计算主要类别
        cam = grad_cam.generate_cam(input, mode_class)

        # 保存叠加图像
        cam_save_path = os.path.join(test_save_path, f"{case}_slice_{ind}_mode_class_{mode_class}_grad_cam_overlay.jpg")
        save_cam_image(cam, slice, cam_save_path)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f'{case}_pred.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f'{case}_img.nii.gz'))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f'{case}_gt.nii.gz'))

    return metric_list


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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        # 确保 input_tensor 需要梯度
        input_tensor.requires_grad = True
        
        # 执行前向传播
        output = self.model(input_tensor)
        
        # 确保 model zero_grad 是在 output 计算之前
        self.model.zero_grad()

        # 创建 one_hot_output
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][class_idx] = 1
        
        # 进行反向传播
        output.backward(gradient=one_hot_output)

        # 获取梯度和激活
        gradients = self.gradients.cpu().numpy()
        activations = self.activation.cpu().numpy()

        # 计算权重
        weights = np.mean(gradients, axis=(2, 3))

        # 初始化 cam
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        # 处理 cam
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        return cam

def save_cam_image(cam, original_image, save_path):
    # 假设 original_image 是 numpy 数组 (slice)
    original_image = np.uint8((original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image)) * 255)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # 将灰度图转为伪彩色图

    # 调整 cam 大小与原图相匹配
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # 创建自定义颜色映射查找表 (LUT)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            lut[i, 0, 0] = 2 * i  # 红色通道
            lut[i, 0, 1] = 2 * i  # 绿色通道
            lut[i, 0, 2] = 0  # 蓝色通道
        else:
            lut[i, 0, 0] = 0  # 红色通道
            lut[i, 0, 1] = 0  # 绿色通道
            lut[i, 0, 2] = 255 - i  # 蓝色通道

    # 生成伪彩色热图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), lut)

    # 将热图叠加在原始图像上
    overlayed_image = cv2.addWeighted(heatmap, 0.4, original_image, 0.6, 0)  # 调整权重以改变效果

    # 保存叠加后的图像
    cv2.imwrite(save_path, np.uint8(overlayed_image))



