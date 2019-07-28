# pytorch-CycleGAN
## Description
CycleGANのpytorch実装

### CycleGAN
[papaer link](https://arxiv.org/abs/1703.10593)
- スタイル変換を行うGAN
- トレーニングデータは完全に対応するペア画像じゃなくても，ドメインの対応さえあっていればよい
- Generator, Discriminatorを各2つもつ

#### Loss
2つのドメイン間のマッピング関数
<img src="https://latex.codecogs.com/gif.latex?G:X&space;\rightarrow&space;Y,&space;F:Y&space;\rightarrow&space;X">
- Adversarial Loss
  - 通常のGANのLoss
<img src="https://latex.codecogs.com/gif.latex?\min_{G}\max_{D_Y}\mathcal{L}_{GAN}(G,&space;D_Y,&space;X,&space;Y)&space;=&space;\mathbb{E}_{y&space;\sim&space;p_{data}(y)}[&space;\log{D_Y(y)}]&space;&plus;&space;\mathbb{E}_{x&space;\sim&space;p_{data}(x)}[\log{1-D_Y(G(x))}]">
<img src="https://latex.codecogs.com/gif.latex?\min_{F}\max_{D_X}\mathcal{L}_{GAN}(F,&space;D_X,&space;X,&space;Y)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{data}(x)}[&space;\log{D_X(x)}]&space;&plus;&space;\mathbb{E}_{y&space;\sim&space;p_{data}(y)}[\log{1-D_X(F(y))}]">
- Cycle Consistency Loss
<img src="https://latex.codecogs.com/gif.latex?{\mathcal{L}_{cyc}(G,&space;F)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{data}(x)}||F(G(x))&space;-&space;x||_1&space;&plus;&space;\mathbb{E}_{y&space;\sim&space;p_{data}(y)}||G(F(y))&space;-&space;y||_1&space;}">
- Full Objective
  - Adversarial LossとCycle Consistency Lossを合わせた最終的なロス
<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(G,&space;F,&space;D_X,&space;D_Y)&space;=&space;\mathcal{L}_{GAN}(G,&space;D_Y,&space;X,&space;Y)&space;&plus;&space;\mathcal{L}_{GAN}(F,&space;D_X,&space;Y,&space;X)&space;&plus;&space;\lambda&space;\mathcal{L}_{cyc}(G,&space;F)">


## Example
### loss


## Requirement
- Python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Click

## Usage
### Training
```
$ bash download_dataset.sh [dataset name]
# dataset name is available following list:
# apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos

$ pip install -r requirements.txt 
$ python main.py train [--dataset]
# training log saved at ./samples/fake_images-[epoch].png
```
