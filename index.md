---
layout: default
title: Rerun Web Viewer
---

<head>
    ...
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>

# Model Under Test

| MiDaS Model          | DIW WHDR | Eth3d AbsRel | Sintel AbsRel | TUM δ1  | KITTI δ1  | NYUv2 δ1 | Imp. % | Par. M | FPS  |
|----------------------|----------|--------------|---------------|---------|-----------|-----------|--------|--------|------|
| **Inference height 224** |          |              |               |         |           |           |        |        |      |
| v3.1 LeViT<sub>224</sub>  | 0.1314   | 0.1206       | 0.31        | 18.21   | 15.27*    | 8.64*     | -40     | 51    | 73  |

# Project Goal

The purpose of this project is to investigate whether or not MIDAS can be used to predict the depth of objects in the KITTI dataset. \
Since MiDas predicts relative depth, we assume that we are able to do the conversion from relative to metric depth given the groundtruth measurements.

# KITTI Dataset

MiDaS is trained using the kitti eigen train split (23158 Images). Therefore, this set must not be part of our evaluation. \
Furthermore, we are interested in determining the 3D object center, since we represent the object depth using the depth measurement at the 3D object center. \
The 3D object center is determined by projecting the 3D pose of the object (from the 3D Object set labels txt files) to the image plane using the projection matrix. 
To map the images from the depth estimation set to the 3D object detection one, we use the mapping given by [train_mapping.txt & train_rand.txt](https://github.com/bostondiditeam/kitti/tree/master/resources/devkit_object/mapping)

Since we are interested in evaluating the objects depth we use the 3D Object Detection labels. 

[KITTI Depth Estimation Dataset](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) \
[KITTI 3D Object Detection Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
$
\textbf{KITTI Depth Estimation Dataset:} \\
\text{Training Set: } D_{\text{train}} = 85898 \text{ Images}\\
\text{Validation Set: } D_{\text{val}} = 6852 \text{ Images}\\
\text{Total Images: } D = D_{\text{train}} \cup D_{\text{test}} = 92750 \text{ Images} \\
\text{Eigen Train Split: } D_{\text{EigenTrainSplit}} = 23158 \text{ Images} \\

\textbf{KITTI 3D Object Detection Dataset:} \\
\text{Training Set: } D_{\text{train3DObject}} = 7481 \text{ Images}\\

\textbf{Dataset Under Test:} \\

\text{Test Set: } D_{\text{Test}} = (D \setminus D_{\text{EigenTrainSplit}}) \cap D_{\text{train3DObject}} = 4255 \text{ Images} \\
$

Our test is given [here](https://drive.google.com/file/d/1ITTkj25Jte3Oc1OyrIViZkvvk04XMIMY/view?usp=drive_link)

# Results

<img src="./images/image1.jpg" width="200" style="display:inline-block;"/>
<img src="./images/image2.jpg" width="200" style="display:inline-block;"/>
<img src="./images/image3.jpg" width="200" style="display:inline-block;"/>

# Rerun Web Viewer Integration

This page displays a live Rerun web viewer that is running on a server.

<!-- Embed the Rerun viewer using an iframe -->
<iframe src="http://localhost:9090?url=ws://localhost:9877" width="100%" height="600" frameborder="0" allowfullscreen></iframe>

> Make sure the Rerun server is running and accessible at the provided URL.




## How to Use

To visualize your data:
1. Start the Rerun server using the following command:
   ```bash
   rerun --serve

http://localhost:9090?url=ws://localhost:9877


