import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwmaUnet as swma_unet
from trainer import trainer_synapse, trainer_ACDC,trainer_ISIC
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='/home/ljc/source/data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument("--volume_path",default='/home/ljc/source/data/ACDC/test', help='path/to/dataset/ACDC/test')
parser.add_argument('--n_skip', type=int, default=1, help='using number of skip-connect, default is num')
parser.add_argument("--z_spacing", default=10)
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
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
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--model_name', type=str,
                    default='SWMA-UNet')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")

config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
        },
        'ISIC_2018': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 1,
        },  
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = swma_unet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)
    
    from thop import profile
    total_params = sum(p.numel() for p in net.parameters())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(1, 3, 224, 224).to(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计
        net = net.to(device)
        _ = net(input)
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # 已分配显存 (GB)
        memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # 已保留显存 (GB)
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # 最大分配显存 (GB)
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3  # 最大保留显存 (GB)

        print(f'GPU Memory Allocated: {memory_allocated:.2f} GB')
        print(f'GPU Memory Reserved: {memory_reserved:.2f} GB')
        print(f'Max GPU Memory Allocated: {max_memory_allocated:.2f} GB')
        print(f'Max GPU Memory Reserved: {max_memory_reserved:.2f} GB')
    flops, params = profile(net, inputs=(input,))
    total_params_m = total_params / 1e6  # 转换为百万
    flops_billion = flops / 1e9  # 转换为亿

    print(f'Total parameters: {total_params_m:.2f}M')
    print(f'FLOPs: {flops_billion:.2f}G')

    # for name, module in net.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         print("@@@@",name)
    
    trainer = {'Synapse': trainer_synapse, 'ACDC': trainer_ACDC,'ISIC_2018': trainer_ISIC}
    trainer[dataset_name](args, net, args.output_dir)




"""
        @@@@ swma_unet.patch_embed.proj
        @@@@ swma_unet.layers.0.blocks.0.attn2.proj1
        @@@@ swma_unet.layers.0.blocks.0.attn2.conv1
        @@@@ swma_unet.layers.0.blocks.0.attn2.conv
        @@@@ swma_unet.layers.0.blocks.0.attn2.proj2
        @@@@ swma_unet.layers.0.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers.0.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers.0.blocks.0.attn2.proj
        @@@@ swma_unet.layers.0.blocks.1.attn2.proj1
        @@@@ swma_unet.layers.0.blocks.1.attn2.conv1
        @@@@ swma_unet.layers.0.blocks.1.attn2.conv
        @@@@ swma_unet.layers.0.blocks.1.attn2.proj2
        @@@@ swma_unet.layers.0.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers.0.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers.0.blocks.1.attn2.proj
        @@@@ swma_unet.layers.1.blocks.0.attn2.proj1
        @@@@ swma_unet.layers.1.blocks.0.attn2.conv1
        @@@@ swma_unet.layers.1.blocks.0.attn2.conv
        @@@@ swma_unet.layers.1.blocks.0.attn2.proj2
        @@@@ swma_unet.layers.1.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers.1.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers.1.blocks.0.attn2.proj
        @@@@ swma_unet.layers.1.blocks.1.attn2.proj1
        @@@@ swma_unet.layers.1.blocks.1.attn2.conv1
        @@@@ swma_unet.layers.1.blocks.1.attn2.conv
        @@@@ swma_unet.layers.1.blocks.1.attn2.proj2
        @@@@ swma_unet.layers.1.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers.1.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers.1.blocks.1.attn2.proj
        @@@@ swma_unet.layers.2.blocks.0.attn2.proj1
        @@@@ swma_unet.layers.2.blocks.0.attn2.conv1
        @@@@ swma_unet.layers.2.blocks.0.attn2.conv
        @@@@ swma_unet.layers.2.blocks.0.attn2.proj2
        @@@@ swma_unet.layers.2.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers.2.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers.2.blocks.0.attn2.proj
        @@@@ swma_unet.layers.2.blocks.1.attn2.proj1
        @@@@ swma_unet.layers.2.blocks.1.attn2.conv1
        @@@@ swma_unet.layers.2.blocks.1.attn2.conv
        @@@@ swma_unet.layers.2.blocks.1.attn2.proj2
        @@@@ swma_unet.layers.2.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers.2.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers.2.blocks.1.attn2.proj
        @@@@ swma_unet.layers.3.blocks.0.attn2.proj1
        @@@@ swma_unet.layers.3.blocks.0.attn2.conv1
        @@@@ swma_unet.layers.3.blocks.0.attn2.conv
        @@@@ swma_unet.layers.3.blocks.0.attn2.proj2
        @@@@ swma_unet.layers.3.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers.3.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers.3.blocks.0.attn2.proj
        @@@@ swma_unet.layers.3.blocks.1.attn2.proj1
        @@@@ swma_unet.layers.3.blocks.1.attn2.conv1
        @@@@ swma_unet.layers.3.blocks.1.attn2.conv
        @@@@ swma_unet.layers.3.blocks.1.attn2.proj2
        @@@@ swma_unet.layers.3.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers.3.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers.3.blocks.1.attn2.proj
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.proj1
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.conv1
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.conv
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.proj2
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.1.blocks.0.attn2.proj
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.proj1
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.conv1
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.conv
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.proj2
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.1.blocks.1.attn2.proj
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.proj1
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.conv1
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.conv
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.proj2
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.2.blocks.0.attn2.proj
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.proj1
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.conv1
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.conv
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.proj2
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.2.blocks.1.attn2.proj
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.proj1
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.conv1
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.conv
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.proj2
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.3.blocks.0.attn2.proj
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.proj1
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.conv1
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.conv
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.proj2
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.ema.conv1x1
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.ema.conv3x3
        @@@@ swma_unet.layers_up.3.blocks.1.attn2.proj
        @@@@ swma_unet.output
    
    
    """
