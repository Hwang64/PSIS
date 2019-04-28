# PSIS
Data Augmentation for Object Detection via Progressive and Selective Instance-Switching

We proposed a simple yet effective data augmentation for object detection, whose core is a progressive and selective instance-switching (PSIS) method for synthetic image generation. The proposed PSIS as data augmentation for object detection benefits several merits, i.e., increase of diversity of samples, keep of contextual coherence in the original images, no requirement of external datasets, and  consideration of instance balance and class importance. Experimental results demonstrate the effectiveness of our PSIS against the existing data augmentation, including horizontal flipping and training time augmentation for FPN, segmentation masks and training time augmentation for Mask R-CNN, multi-scale training strategy for SNIPER, and Context-DA for BlitzNet. The improvement on both object detection and instance segmentation tasks suggest our proposed PSIS has the potential to improve the performance of other applications (i.e., keypoint detection), which will be investigated in future work.

# #PSIS Framework
<a href="https://www.codecogs.com/eqnedit.php?latex=Overview&space;of&space;instance-switching&space;strategy&space;for&space;image&space;generation.&space;We&space;first&space;select&space;a&space;candidate&space;set&space;$\Omega^{c}_{cad}$&space;from&space;all&space;training&space;data&space;$\Omega^{c}_{all}$&space;for&space;each&space;class&space;based&space;on&space;shape&space;and&space;scale&space;of&space;instance.&space;For&space;a&space;quadruple&space;$\{I_{A}^{c}$,$I_{B}^{c},i_{A}^{c},i_{B}^{c}\}$&space;in&space;$\Omega^{c}_{cad}$,&space;we&space;switch&space;the&space;instances&space;$i_{A}^{c}$&space;and&space;$i_{B}^{c}$&space;through&space;rescaling&space;and&space;cut-paste.&space;Finally,&space;Gaussian&space;blurring&space;is&space;used&space;to&space;smooth&space;the&space;boundary&space;artifacts.&space;Thus,&space;synthetic&space;images&space;$\{\hat{I}_{A}^{c}$,$\hat{I}_{B}^{c}\}$&space;are&space;generated." target="_blank"><img src="https://latex.codecogs.com/gif.latex?Overview&space;of&space;instance-switching&space;strategy&space;for&space;image&space;generation.&space;We&space;first&space;select&space;a&space;candidate&space;set&space;$\Omega^{c}_{cad}$&space;from&space;all&space;training&space;data&space;$\Omega^{c}_{all}$&space;for&space;each&space;class&space;based&space;on&space;shape&space;and&space;scale&space;of&space;instance.&space;For&space;a&space;quadruple&space;$\{I_{A}^{c}$,$I_{B}^{c},i_{A}^{c},i_{B}^{c}\}$&space;in&space;$\Omega^{c}_{cad}$,&space;we&space;switch&space;the&space;instances&space;$i_{A}^{c}$&space;and&space;$i_{B}^{c}$&space;through&space;rescaling&space;and&space;cut-paste.&space;Finally,&space;Gaussian&space;blurring&space;is&space;used&space;to&space;smooth&space;the&space;boundary&space;artifacts.&space;Thus,&space;synthetic&space;images&space;$\{\hat{I}_{A}^{c}$,$\hat{I}_{B}^{c}\}$&space;are&space;generated." title="Overview of instance-switching strategy for image generation. We first select a candidate set $\Omega^{c}_{cad}$ from all training data $\Omega^{c}_{all}$ for each class based on shape and scale of instance. For a quadruple $\{I_{A}^{c}$,$I_{B}^{c},i_{A}^{c},i_{B}^{c}\}$ in $\Omega^{c}_{cad}$, we switch the instances $i_{A}^{c}$ and $i_{B}^{c}$ through rescaling and cut-paste. Finally, Gaussian blurring is used to smooth the boundary artifacts. Thus, synthetic images $\{\hat{I}_{A}^{c}$,$\hat{I}_{B}^{c}\}$ are generated." /></a>
<img src="https://github.com/Hwang64/PSIS/blob/master/img/pipeline.jpg">

## Synthetic Image Generation


## Results

### Appling PSIS to State-of-the-art Detectors
We directly employ this dataset to train four state-of-the-art detectors (i.e., [FPN](https://github.com/open-mmlab/mmdetection) , [Mask R-CNN](https://github.com/open-mmlab/mmdetection) , [BlitzNet](https://github.com/dvornikita/blitznet) and [SNIPER](https://github.com/mahyarnajibi/SNIPER)), and report results on test server for comparing with other augmentation methods.

### FPN
$\times 2$ means two times training epochs, which is regarded as training-time augmentation.Above results clearly demonstrate our PSIS is superior and complementary to horizontal flipping and training-time augmentation methods.
Networks | Avg.Precision,IOU: | Avg.Precision,Area: |  Avg.Recal,#Det:  |    Avg.Recal,Area:  | 
|--------|:--------------------:|:---------------------:|:-------------------:|:---------------------:|
|--------|0.5:0.95 | 0.50 | 0.75| Small |  Med. | Large |   1  |  10   |  100 | Small |  Med. | Large |
|:------:|:-------:|:----:|:---:|:-----:|:-----:|:-----:|:----:|:-----:|:----:|:-----:|:-----:|:-----:|
|   ori  |  38.6   | 60.4 | 41.6|  22.3 | 42.8  |  50.0 | 31.8 | 50.6  | 53.2 |  34.5 |  57.7 |  66.8 |
|  psis  |  39.8   | 61.0 | 43.4|  22.7 | 44.2  |  52.1 | 32.6 | 51.1  | 53.6 |  34.8 |  59.0 |  68.5 |
| ori$times$2  |  39.4   | 60.7 | 43.0 |  21.1  |  43.6 |  52.1 | 32.5 | 51.0  | 53.4 |  33.6 |  57.6 |  68.6 |
| psis$times$2 |  40.2   | 61.1 | 44.2 |  22.3  |  45.7 |  51.6 | 32.6 | 51.2  | 53.6 |  33.6 |  58.9 |  68.8 |

### Mask R-CNN

### BlitzNet

### SNIPER

### Examples of Synthetic Images Generated by our IS
Here we show some examples of synthetic images generated by our IS strategy. The new (switched) instances are denoted in red boxes, and our instance-switching strategy can clearly preserve contextual coherence in the original images.
<img src="https://github.com/Hwang64/PSIS/blob/master/img/examples.jpg">

## Acknowledgments 
Some codes are referenced from [syndata-generation](https://github.com/debidatta/syndata-generation). Thanks for their codes!
