# CNN + Transformer

## TODO

### 1. ResNet 50, 101 train / test able code (Mnist, COCO)

### 2. Transformer trainable code (Mnist)

### 3. ResNet + Transformer 



# Usage

An example of training/testing ResNet50 or ResNet101

CUDA 11.0 with CUDNN 8.2.0


### 0) Prepare Python 3.7 virual environments and set requirements.txt
```bash
conda create -n [env name] python=3.7 -y
git clone https://github.com/wlgjs8/Transformer
cd Transformer
pip install -r requirements.txt
```

### 1) Train Model on MNIST Dataset
```bash
python example/train_resnet.py -net resnet50 
```

### 2) Test Model on MNIST Dataset
```bash
python example/test_resnet.py -net resnet50 -weight [path to checkpoint]
```
