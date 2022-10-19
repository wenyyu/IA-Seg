# Improving Nighttime Driving-Scene Segmentation via Dual Image-adaptive Learnable Filters
####  [[arxiv]]([https://arxiv.org/abs/2207.01331]) 
Wenyu Liu, Wentong Li, [Jianke Zhu](https://person.zju.edu.cn/jkzhu/645901.html), Miaomiao Cui, Xuansong Xie, [Lei Zhang](https://web.comp.polyu.edu.hk/cslzhang/)
## Requirements
* python3.7
* pytorch==1.5.0
* cuda10.2
* scikit-image
* opencv-python
## Datasets and Models
**Cityscapes**:  [Cityscape](https://www.cityscapes-dataset.com/) 
**NightCity**:  [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html/) 
**ACDC**:  [ACDC](https://acdc.vision.ee.ethz.ch/) 
**Dark-Zurich**: [Dark-Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)  

**Models**: [Google Drive](https://drive.google.com/drive/folders/1UDxIAk8v56455XfTB52J2jmgcZEtLAbH?usp=sharing/) 

## Test

```
Step1: download the [Models](https://drive.google.com/drive/folders/1UDxIAk8v56455XfTB52J2jmgcZEtLAbH?usp=sharing) and put it in the root.
Step2: change the data and model paths in configs/test_config.py
Step3: run "python evaluation_supervised.py" for supervised experiments,  "python evaluation_unsupervised.py" for unsupervised experiments,
Step4: run "python compute_iou.py"
```

## Training 
```
Step1: download the [pre-trained models](https://www.dropbox.com/s/3n1212kxuv82uua/pretrained_models.zip?dl=0) and put it in the root.
Step2: change the data and model paths in configs/train_config.py
Step3: run "python train_unsupervised.py" for unsupervised experiments, run "python train_nightcity.py" for supervised nightcity experiments, run "python train_acdc_night.py" for supervised acdc experiments
```
## Acknowledgments
The code is based on DANNet, PSPNet, Deeplab-v2 and RefineNet.
## More works
The image-adaptive filtering techniques used in the detection task can be found in our AAAI2022 paper.

### Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions [[Link]](https://github.com/wenyyu/Image-Adaptive-YOLO)

## Citation

```shell

@article{liu2022improving,
  title={Improving Nighttime Driving-Scene Segmentation via Dual Image-adaptive Learnable Filters},
  author={Liu, Wenyu and Li, Wentong and Zhu, Jianke and Cui, Miaomiao and Xie, Xuansong and Zhang, Lei},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2022}
}
```
