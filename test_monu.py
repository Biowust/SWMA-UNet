import torch.optim
from datasets.dataset_Monu import RandomGenerator,ValGenerator,ImageToImage2D,train_one_epoch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import *
import cv2
from pathlib import Path
from networks.vision_transformer import SwmaUnet as swma_unet
import argparse
from config import get_config
from GradCAM import show_cam_on_image
from PIL import Image
from skimage import morphology
from GradCAM import GradCAM


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
  
    predict_save = cv2.pyrUp(predict_save,(448,448))
    predict_save = cv2.resize(predict_save,(2000,2000))
    cv2.imwrite(save_path,predict_save * 255)

    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_img = input_img.to(device)
    
    model.eval()

    output = model(input_img)
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (args.img_size, args.img_size))

    save_path = vis_save_path.rsplit('.', 1)[0] + '_predict.jpg'
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=save_path)

    cam = GradCAM(model=model, target_layers=[model.swma_unet.output])
    output = output.int()
    mode_class = torch.mode(torch.flatten(output)).values.item()
    grayscale_cam = cam(input_tensor=input_img, target_category=mode_class)  
    grayscale_cam = grayscale_cam[0, :]

    if img_RGB is None:
        # 从 input_img 转换为 RGB 图像
        img_RGB = input_img[0].permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
        img_RGB = (img_RGB - np.min(img_RGB)) / (np.max(img_RGB) - np.min(img_RGB)) 

    if np.max(img_RGB) > 1.0:  
        img_RGB = img_RGB / 255.0
    cam_image = show_cam_on_image(img_RGB, grayscale_cam, use_rgb=True)

    heatmap_save_path = vis_save_path.rsplit('.', 1)[0] + '_heatmap.jpg'
    os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
    save_cam_heatmap(cam_image, grayscale_cam, heatmap_save_path)


    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_num = 14

    model_path = '/home/ljc/source/outputs/swmaUnet/Monu/2024-11-16/epoch_437_0.8043757081031799.pth'
    vis_path = '/home/ljc/source/outputs/swmaUnet/Monu/2024-11-16/pic/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                    default='/home/ljc/source/data/MoNuSeg/Val_Folder/', help='root dir for validation volume data')
    parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')     
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--cfg', type=str, default='/home/ljc/source/SWMA-UNet/configs/swin_tiny_patch4_window7_224_lite.yaml',metavar="FILE", help='path to config file', )    
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )      
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--batch_size', type=int, default=36,help='batch_size per gpu') 
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--model_name', type=str,default='SwmaUnet')                
    args = parser.parse_args()

    config = get_config(args)
    checkpoint = torch.load(model_path, map_location='cuda')
    model = swma_unet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[args.img_size, args.img_size])
    test_dataset = ImageToImage2D(args.test_path, tf_test,image_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    list=[]
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255

            # 使用原图名字作为文件名
            original_name = names[0]  # 假设 `names` 是一个包含文件名的列表
            base_name = Path(original_name).stem  # 去掉路径和扩展名，保留纯文件名

            # 保存分割图
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = args.img_size, args.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            save_path = Path(vis_path) / f"{base_name}_lab.jpg"  # 保存路径
            plt.savefig(save_path, dpi=300)
            plt.close()

            # 处理预测和可视化
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(
                model, input_img, None, lab,
                str(Path(vis_path) / base_name),  # 将 Path 对象转为字符串
                dice_pred=dice_pred, dice_ens=dice_ens
            )
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            list.append(dice_pred_t)
            torch.cuda.empty_cache()
            pbar.update()

    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    print(list)




