# Digital Image Process Final Project Group 18

B07902072 陳光裕
R09922129 詹鈞凱

Official [PyTorch](https://pytorch.org/) implementation of the paper:

[**ArtFlow: Unbiased Image Style Transfer via Reversible Neural Flows**](https://arxiv.org/abs/2103.16877)  
[Jie An<sup>*</sup>](https://www.cs.rochester.edu/u/jan6/), [Siyu Huang<sup>*</sup>](https://siyuhuang.github.io), [Yibing Song](https://ybsong00.github.io), [Dejing Dou](https://ix.cs.uoregon.edu/~dou/), [Wei Liu](https://sse.cuhk.edu.cn/en/faculty/liuwei) and [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)  
CVPR 2021

ArtFlow is a universal style transfer method that consists of reversible neural flows and an unbiased feature transfer module. ArtFlow adopts a projection-transfer-reversion scheme instead of the encoder-transfer-decoder to avoid the content leak issue of existing style transfer methods and consequently achieves unbiased style transfer in continuous style transfer.

## Dependencies

* Python=3.6
* PyTorch=1.8.1
* CUDA=10.2
* cuDNN=7.6
* Scipy=1.5.2
* [FFmpeg](https://ffmpeg.org/download.html)

If you are a `conda` user, you can execute the following command in the directory of this repository to create a new environment with all dependencies installed.
```
conda env create -f environment.yaml
```

**You should execute code under `code/` directory.**

## Pretrained Models

```
CUDA_VISIBLE_DEVICES=0 python3 -u test.py --content_dir ../data/content_demo --style_dir ../data/style_demo --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result
```

If you are in windows anaconda system, use
```
set CUDA_VISIBLE_DEVICES=0
python3 -u test.py --content_dir ../data/content_demo --style_dir ../data/style_demo --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result
```
* `content_dir` : path for the content images. Default is `../data/content_demo`
* `style_dir` : path for the style images. Default is `../data/style_demo`.
* `size` : image size for style transfer. Default is `256`.
* `n_flow` : number of the flow module used per block in the backbone network. Default is `8`.
* `n_block` : number of the block used in the backbone network. Default is `2`.
* `operator` : style transfer module. Options: `[adain, wct, decorator]`.
* `decoder` : path for the pre-trained model, if you let the `--operator wct`, then you should load the pre-trained model with `--decoder experiments/ArtFlow-WCT/glow.pth`. Otherwise, if you use AdaIN, you should set `--decoder experiments/ArtFlow-AdaIN/glow.pth`. If you want to use this code for portrait style transfer, please set `--operator adain` and `--decoder experiments/ArtFlow-AdaIN-Portrait/glow.pth`.
* `output` : path of the output directory. This code will produce a style transferred image for every content-style combination in your designated directories. Default is `../data/result`

## Task

### Images

```
set CUDA_VISIBLE_DEVICES=0
python3 -u test.py --content_dir ../data/task/image_0_pic --style_dir ../data/style_demo --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task
python3 -u test.py --content_dir ../data/task/image_0_pic --style_dir ../data/style_example --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task

python3 -u test.py --content_dir ../data/task/image_1_pic --style_dir ../data/style_demo --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task
python3 -u test.py --content_dir ../data/task/image_1_pic --style_dir ../data/style_example --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task
```

### Videos

```
set CUDA_VISIBLE_DEVICES=0
mkdir video_0_pic
mkdir video_1_pic
ffmpeg -i ../data/task/video_0 video_0_pic/pic%04d.png
ffmpeg -i ../data/task/video_1 video_1_pic/pic%04d.png

python3 -u test.py --content_dir ../data/task/video_0_pic --style_dir ../data/style_demo --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task/video_0_pic
python3 -u test.py --content_dir ../data/task/video_0_pic --style_dir ../data/style_example --size 256 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../data/result/task/video_0_pic

cd ../data/result/task/video_0_pic
ffmpeg -i pic%04d_stylized_red-canna.jpg -r 60 video_0_stylized_red-canna.mp4
ffmpeg -i pic%04d_stylized_001.jpg -r 60 ../output_0_demo1.mp4
ffmpeg -i pic%04d_stylized_002.jpg -r 60 ../output_0_demo2.mp4
ffmpeg -i pic%04d_stylized_003.jpg -r 60 ../output_0_demo3.mp4

cd ../video_1_pic
ffmpeg -i pic%04d_stylized_red-canna.jpg -r 60 video_1_stylized_red-canna.mp4
ffmpeg -i pic%04d_stylized_001.jpg -r 60 ../output_1_demo1.mp4
ffmpeg -i pic%04d_stylized_002.jpg -r 60 ../output_1_demo2.mp4
ffmpeg -i pic%04d_stylized_003.jpg -r 60 ../output_1_demo3.mp4
```

## Training

We use **Google Colab** to train our model, the code above should run in python environment.

```
import os
import sys
import string
import shutil

# Kaggle API

!mkdir ~/.kaggle
!cp drive/MyDrive/DIP/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download Moeimouto Dataset from Kaggle

!kaggle datasets download -d mylesoneill/tagged-anime-illustrations
!unzip -q tagged-anime-illustrations.zip "moeimouto-faces/moeimouto-faces/*"
!rm tagged-anime-illustrations.zip
!mkdir style_img

# Data Preprocess

#Generate the file paths to traverse, or a single path if a file name was given
def getfiles(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in files:
                yield os.path.join(root, name)
    else:
        yield path

destination = "./style_img/"
fromdir = "./moeimouto-faces/"
for f in getfiles(fromdir):
    filename = f.split('/')[-1]
    if os.path.isfile(destination+filename):
        filename = f.replace(fromdir,"",1).replace("/","_")
    #os.rename(f, destination+filename)
    shutil.move(f, destination+filename)

!rm -rf moeimouto-faces
!find ./style_img/ -name "*.csv" -type f -delete

# Download FFHQ Dataset from Kaggle

!kaggle datasets download -d greatgamedota/ffhq-face-data-set
!unzip -q ffhq-face-data-set.zip
!rm ffhq-face-data-set.zip

# Download Oil Painting Dataset

!kaggle datasets download -d herbulaneum/oil-painting-images
!unzip -q oil-painting-images.zip "oilpainting_256/crop/*"
!cp -R oilpainting_256/crop/ .
!rm -rf oilpainting_256
!rm oil-painting-images.zip
!find ./crop/ -name "*.jpg_flip" -type f -delete
!ls

# Download Code

!git clone https://github.com/pkuanjie/ArtFlow

# Training 
!mkdir models
!cp drive/MyDrive/DIP/vgg_normalised.pth models
!python3 -u ArtFlow/train.py --content_dir thumbnails128x128 --style_dir crop \
         --n_flow 8 --n_block 2 --batch_size 4 --content_weight 0.2 --max_iter 10000 \
         --save_model_interval 2000 --operator adain --save_dir drive/MyDrive/DIP/models_oil  

# Testing

# ArtFlow Trained by Ourself
!mkdir drive/MyDrive/DIP/test/oil_adain_10000
!python3 -u ArtFlow/test.py --content_dir drive/MyDrive/DIP/test/content \
    --style_dir drive/MyDrive/DIP/test/style --size 256 --n_flow 8 --n_block 2 \
    --operator adain --decoder drive/MyDrive/DIP/models_oil/glow_10000.pth \
    --output drive/MyDrive/DIP/test/oil_adain_10000

# ArtFlow Paper's Pre-trained Model

!python3 -u ArtFlow/test.py --content_dir drive/MyDrive/DIP/test/content \
    --style_dir drive/MyDrive/DIP/test/style --size 256 --n_flow 8 --n_block 2 \
    --operator adain --decoder drive/MyDrive/DIP/ArtFlow-AdaIN/glow.pth \
    --output drive/MyDrive/DIP/test/paper_adain

!python3 -u ArtFlow/test.py --content_dir drive/MyDrive/DIP/test/content \
    --style_dir drive/MyDrive/DIP/test/style --size 256 --n_flow 8 --n_block 2 \
    --operator wct --decoder drive/MyDrive/DIP/ArtFlow-WCT/glow.pth \
    --output drive/MyDrive/DIP/test/paper_wct

# Original AdaIN

!git clone https://github.com/naoto0804/pytorch-AdaIN
!mkdir models
!cp drive/MyDrive/DIP/adain/decoder.pth models/
!cp drive/MyDrive/DIP/adain/vgg_normalised.pth models/
!mkdir drive/MyDrive/DIP/test/adain
!python pytorch-AdaIN/test.py --content_dir drive/MyDrive/DIP/test/content \
    --style_dir drive/MyDrive/DIP/test/style --output drive/MyDrive/DIP/test/adain \
    --content_size 512 --style_size 512 --crop
```
