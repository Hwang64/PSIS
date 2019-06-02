# PSIS

Data Augmentation for Object Detection via Progressive and Selective Instance-Switching

## Abstract

We proposed a simple yet effective data augmentation for object detection, whose core is a progressive and selective instance-switching (PSIS) method for synthetic image generation. The proposed PSIS as data augmentation for object detection benefits several merits, i.e., increase of diversity of samples, keep of contextual coherence in the original images, no requirement of external datasets, and  consideration of instance balance and class importance. Experimental results demonstrate the effectiveness of our PSIS against the existing data augmentation, including horizontal flipping and training time augmentation for FPN, segmentation masks and training time augmentation for Mask R-CNN, multi-scale training strategy for SNIPER, and Context-DA for BlitzNet. The experiments are conducted on the challenging MS COCO benchmark, and results demonstrate our PSIS brings clear improvement over various state-of-the-art detectors

## Framework

<img src="https://github.com/Hwang64/PSIS/blob/master/img/pipeline.jpg">

## Machine configurations

- OS: Linux 16.02
- GPU: TiTan 1080 Ti
- CUDA: version 8.0
- CUDNN: version 5.1

Slight changes may not results instabilities

## Synthetic Image Generation

In this part, we provide the code for synthetic image generation by taking MS COCO 2017 training set as benchmark. We first generate the instance masks for the images in the training set. Then we use the methods describe in Section 3.1 in the paper to generate the  quadruple. At last, depending on the quadruple, we generate the synthetic images by switching the instance.

### Instance Mask Generation

Use the code ```extract_mask.m``` to generate instance mask for the images in MS COCO 2017 training dataset.

### Quadruple Generation

Use the code ```extract_annotation_pair.py``` to generate quadruple for each category which satisfy the conditions. The ouput quadruple will saved in a txt file. We also provide the Omega_uni, Omega_equ and Omega_aug in ```dataset``` which follow the instance distribution in the paper.

### Synthetic Image and Annotation Generation

At last, use the code ```instance_switch.py``` to generate the corresponding images depending on the input quadruple. Meanwhile, the corresponding annotation file will also be generated. 

For generting images, just modify the ```ANN2ann_FILE``` in file ```instance_switch```(e.g., ```dataset/omega_uni.txt```) and the synthetic images and annotation file will be generated in the corresponding directory.

Our synthetic images and corresponding annotation files can be downloaded in [Here](https://pan.baidu.com/s/1McDAmN_PttTz69SNN-Q2wg)(Type the Extraction Code: wnjx)

### Class Imbalance Loss

The code for class imbalance loss is in ```\class_imbalance_loss``` directory, please refer to the ```\class_imbalance_loss\README.md``` for detail using.

## Results

### Applying PSIS to State-of-the-art Detectors

We directly employ this dataset to train four state-of-the-art detectors (i.e., [FPN](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) , [Mask R-CNN](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) , [BlitzNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dvornik_BlitzNet_A_Real-Time_ICCV_2017_paper.pdf) and [SNIPER](https://arxiv.org/abs/1805.09300)), and report results on test server for comparing with other augmentation methods.

#### FPN

We adopt PSIS to FPN by the publicly availabel toolkit. The configuration files are in the ```configs/FPN```. For more training and testing information, please refer to the [code](https://github.com/open-mmlab/mmdetection). The results are shown as belows:

|Training Sets | AP@0.50:0.95 | AP@0.50 | AP@0.75| AP@Small | AP@Med. | AP@Large |  AR@1 | AR@10 | AR@100 | AR@Small | AR@Med. | AR@Large  | 
|--------------|:------------:|:-------:|:------:|:--------:|:-------:|:--------:|:-----:|:-----:|:------:|:--------:|:-------:|:----:|
|   ori* |  38.1   | 59.1 | 41.3|  20.7 | 42.0  |  51.1 | 31.6 | 49.3  | 51.5 |  31.1 |  55.7 |  66.7 |
|  psis* |  38.7   | 59.7 | 41.8|  21.6 | 43.0  |  51.7 | 32.0 | 50.0  | 52.3 |  32.3 |  56.4 |  67.6 |
|   ori  |  38.6   | 60.4 | 41.6|  22.3 | 42.8  |  50.0 | 31.8 | 50.6  | 53.2 |  34.5 |  57.7 |  66.8 |
|  psis([model](https://drive.google.com/open?id=1a5W8CWFQWQaw8TqVutQmCQWzwcRf3YKE))  |  39.8   | 61.0 | 43.4|  22.7 | 44.2  |  52.1 | 32.6 | 51.1  | 53.6 |  34.8 |  59.0 |  68.5 |
| ori×2  |  39.4   | 60.7 | 43.0 |  21.1  |  43.6 |  52.1 | 32.5 | 51.0  | 53.4 |  33.6 |  57.6 |  68.6 |
| psis×2 ([Coming Soon]())|  40.2   | 61.1 | 44.2 |  22.3  |  45.7 |  51.6 | 32.6 | 51.2  | 53.6 |  33.6 |  58.9 |  68.8 |

×2 means two times training epochs, which is regarded as training-time augmentation and * indicates no horizontal fliping. Above results clearly demonstrate our PSIS is superior and complementary to horizontal flipping and training-time augmentation methods.

#### Mask R-CNN

We evaluate PSIS using Mask R-CNN. The configuration files are in the ```configs/Mask R-CNN```. For more training and testing information, please refer to the [code](https://github.com/open-mmlab/mmdetection). The results are shown as belows: 

|Training Sets | AP@0.50:0.95 | AP@0.50 | AP@0.75| AP@Small | AP@Med. | AP@Large |  AR@1 | AR@10 | AR@100 | AR@Small | AR@Med. | AR@Large  | 
|--------------|:------------:|:--------:|:------:|:--------:|:-------:|:--------:|:-----:|:-----:|:------:|:--------:|:-------:|:----:|
|   ori  |  39.4   | 61.0 | 43.3 |  23.1  | 43.7  |  51.3 | 32.3 | 51.5  | 54.3 |  34.9 |  58.7 |  68.5 |
|  psis([model](https://drive.google.com/open?id=13kB4zvwR__O9vSGz9cvy4Br3C-mwSTGZ))  |  40.7   | 61.8 | 44.5 |  23.4  | 45.2  |  53.0 | 33.3 | 52.8  | 55.4 |  35.5 |  59.7 |  70.3 |
| ori×2  |  40.4   | 61.6 | 44.2 |  22.3  |  44.8 |  52.9 | 33.1 | 52.0  | 54.5 |  34.7 |  58.8 |  69.5 |
| psis×2([Coming Soon]()) |  41.2   | 62.5 | 45.4 |  23.7  |  46.0 |  53.6 | 33.4 | 52.9  | 55.5 |  36.2 |  60.0 |  70.3 |

×2 means two times training epochs, which is regarded as training-time augmentation. Above results clearly demonstrate our PSIS is superior and complementary to training-time augmentation method.

#### BlitzNet

 We evaluate PSIS with the recently proposed context-based data augmentation method. We adopt PSIS to BlitzNet, For more traning and testing information, please refer to [code](https://github.com/dvornikita/blitznet).

|Training Sets | AP@0.50:0.95 | AP@0.50 | AP@0.75| AP@Small | AP@Med. | AP@Large | 
|--------------|:------------:|:-------:|:------:|:--------:|:-------:|:--------:|
|   ori  |  27.3   | 46.0 | 28.1 |  10.7  | 26.8  |  46.0 | 
|   [Context-DA](https://github.com/dvornikita/context_aug)  |  28.0   | 46.7 | 28.9 |  10.7  | 27.8  |  47.0 |
|  psis([Coming Soon]())  |  30.8   | 50.0 | 32.2 |  12.6  | 31.0  |  50.2 | 

#### SNIPER

We use SNIPER to verify the effectiveness of PSIS under multi-scale training strategy. The configuration files are in the ```configs/SNIPER```. For more training and testing information, please refer to the [code](https://github.com/mahyarnajibi/SNIPER). The results are shown as belows: 

|Training Sets | AP@0.50:0.95 | AP@0.50 | AP@0.75| AP@Small | AP@Med. | AP@Large |  AR@1 | AR@10 | AR@100 | AR@Small | AR@Med. | AR@Large  | 
|--------------|:------------:|:-------:|:------:|:--------:|:-------:|:--------:|:-----:|:-----:|:------:|:--------:|:-------:|:----:|
|   ori  |  43.4   | 62.8 | 48.8 |  27.4  | 45.2  |  56.2 | N/A | N/A  | N/A |  N/A |  N/A |  N/A |
|  psis([Coming Soon]())  |  44.2   | 63.5 | 49.3 |  29.3  | 46.2  |  57.1 | 35.0 | 60.1  | 65.9 |  50.4 |  70.4 |  78.0 |

### Generalization to Instance Segmentation

We verify the generalization ability of our PSIS on instance segmentation task of MS COCO 2017. The instance segmetatnion results are shown belows:

|Training Sets | AP@0.50:0.95 | AP@0.50 | AP@0.75| AP@Small | AP@Med. | AP@Large |  AR@1 | AR@10 | AR@100 | AR@Small | AR@Med. | AR@Large  | 
|--------------|:------------:|:-------:|:------:|:--------:|:-------:|:--------:|:-----:|:-----:|:------:|:--------:|:-------:|:----:|
|   ori  |  35.9   | 57.7 | 38.4 |  19.2  | 39.7  |  49.7 | 30.5 | 47.3  | 49.6 |  29.7 |  53.8 |  65.8 |
|  psis([model](https://drive.google.com/open?id=13kB4zvwR__O9vSGz9cvy4Br3C-mwSTGZ))  |  36.7   | 58.4 | 39.4 |  19.0  | 40.6  |  50.2 | 31.0 | 48.2  | 50.3 |  29.8 |  54.4 |  66.9 |
| ori×2  |  36.6   | 58.2 | 39.2 |  18.5  |  40.3 |  50.4 | 31.0 | 47.7  | 49.7 |  29.5 |  53.5 |  66.6 |
| psis×2([Coming Soon]()) |  37.1   | 58.8 | 39.9 |  19.3  |  41.2 |  50.8 | 31.1 | 47.7  | 50.4 |  30.2 |  54.5 |  67.9 |

Above results clearly show PSIS offers a new and complementary way to use instance masks for improving both detection and segmentation performance.

## Examples of Synthetic Images Generated by our IS

Here we show some examples of synthetic images generated by our IS strategy. The new (switched) instances are denoted in red boxes, and our instance-switching strategy can clearly preserve contextual coherence in the original images.

<img src="https://github.com/Hwang64/PSIS/blob/master/img/examples.jpg">
