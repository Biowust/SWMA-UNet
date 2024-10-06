
# SWMA-UNet

This repository is the official implementation of SWMA-UNet. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## DataSet
You can download four processed datasets: ACDC, Synapse, ISIC 2018, ISIC 2017. [click this](https://zenodo.org/records/13741332)

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --dataset <dataset name> --cfg /configs/swin_tiny_patch4_window7_224_lite.yaml --root_path <dataset path> --max_epochs 500 --output_dir <output dir> --img_size 224 --base_lr 0.0001 --batch_size 36 
```


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python test.py --cfg /configs/swin_tiny_patch4_window7_224_lite.yaml --volume_path data/Synapse  --num_classes 9 --list_dir lists/lists_Synapse --output_dir <output dir> --model_weight <weigth path> --max_epochs 500 --img_size 224 --is_savenii --test_save_dir <save dir> --model_name XXX
```



## Pre-trained Models

You can download pretrained models here:

- [swin transformer model](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY) 





## References
[TransUNet](https://github.com/Beckschen/TransUNet)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[SwinUNet](https://github.com/HuCaoFighting/Swin-Unet?tab=readme-ov-file)

