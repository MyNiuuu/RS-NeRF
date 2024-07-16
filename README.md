  <h2> 🎢 RS-NeRF: Neural Radiance Fields from Rolling Shutter Images (ECCV 2024) </h2>
<div>
    <a href='https://myniuuu.github.io/' target='_blank'>Muyao Niu</a> <sup></sup> &nbsp;
    <a href='' target='_blank'>Tong Chen</a><sup></sup> &nbsp;
    <a href='' target='_blank'>Yifan Zhan</a><sup></sup> &nbsp;
    <a href=''>Zhuoxiao Li</a><sup></sup> &nbsp; 
    <a href='' target='_blank'>Xiang Ji</a><sup></sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=en' target='_blank'>Yinqiang Zheng</a><sup>*</sup> &nbsp;
</div>
<div>
    The University of Tokyo &nbsp; <sup>*</sup> Corresponding Author &nbsp; 
</div>
<a href='https://arxiv.org/abs/2407.10267'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a>


In *European Conference on Computer Vision (ECCV) 2024*

---

Stay tuned. Feel free to contact me for bugs or missing files.


## Setup Procedures

### Python Environment

```
conda create -n rsnerf python==3.10
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Dataset

We contribute synthetic and real datasets for evaluating RS-related novel-view synthesis techniques that follows the forward-facing manner. 

Download the synthetic and real dataset from this [link](https://drive.google.com/drive/folders/1xyr_lSex5XZjIMH3mOAnKYoXheywlhWt?usp=sharing) and unzip them to the current directory.

### Pretrained RAFT model for multi-sampling

Download the pretrained RAFT model (`raft-things.pth`) from this [link](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing) and unzip it to `./raft_models`.

## Training

### Synthetic dataset

```
python train.py \
--config configs/wine.txt
```

### Real dataset

```
python train_real.py \
--config configs/real_toy.txt
```


## Acknowledgments

We appreciate for [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF), upon which we build our code implementation. We would also appreciate the code release of [USB-NeRF](https://github.com/WU-CVGL/USB-NeRF), [rspy](https://github.com/DelinQu/rspy), [JAMNet](https://github.com/GitCVfb/JAMNet), [CVR](https://github.com/GitCVfb/CVR), and [DeepUnroll](https://github.com/ethliup/DeepUnrollNet) for comparison and evaluation. 
