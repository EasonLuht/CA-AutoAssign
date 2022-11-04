# CA-AutoAssign: An Anchor-free Defect Detector for complex background Based on Pixel-Wise Adaptive Multi-Scale Feature Fusion
![The schematic of the AGM design details](./images/AGM.jpg)



| Method                 | Backone−Neck       | AP   | AP50 | AP75 | APS  | APM  | APL  |
|------------------------|----------------|------|------|------|------|------|------|
| Anchor-base:    |                        |       |       |       |       |       |       |
| RetinaNet       | ResNet-50-FPN          | 31.4  | 64.5  | 21.0  | 11.9  | 18.2  | 33.0  |
| YOLOF           | ResNet-50-DilatedEncoder  | 32.6  | 65.2  | 25.5  | 8.7   | 15.5  | 35.6  |
| Faster R-CNN    | ResNet-50-FPN          | 35.5  | 69.7  | 25.7  | 12.3  | 21.3  | 35.2  |
| Cascade R-CNN   | ResNet-50-FPN          | 40.1  | 72.8  | 33.1  | 17.6  | 21.2  | 39.6  |
| Sparse R-CNN    | ResNet-50-FPN          | 40.9  | 75.5  | 33.6  |  17.0  |  31.4  | 43.8  |
| Deformable DETR    | Encoder−Decoder          | 39.9  | 72.1  | 32.7  | 13.4  | 29.1  | 42.9  |
| ES-Net    |     -     | 41.1   | 76.2  | 33.7  | -  | -  | -  |
| Anchor-free:    |                        |       |       |       |       |       |       |
| FCOS            | ResNet-101-FPN         | 35.3  | 69.0  | 28.5  | 13.4  | 20.6  | 35.0  |
| YOLOX           | CSPDarknet-YOLOXPAFPN  | 35.0  | 68.1  | 28.2  | 21.6  | 18.7  | 35.9  |
| RepPoints          | ResNet-50−FPN  | 33.3  | 69.0  | 25.0  |  15.4  | 20.3  |  30.3  |
| TOOD          | ResNet-50−FPN  | 40.4  | 74.5  | 32.9  |  16.0   | 22.1  |  40.3  |
| AutoAssign      | ResNet-50-FPN          | 41.5  | 79.3  | 33.2  | 23.1  | 27.7  | 39.8  |
| AutoAssign      | Swin-Tiny-FPN          | 45.2  | 81.3  | 37.9  | 21.6  | 35.3  | 51.0  |
| CAL-AutoAssign  | Swin-Tiny-CAL          | 53.1  | 89.1  | 49.8  | 25.7  | 40.7  | 64.5  |


[权重文件下载](https://drive.google.com/file/d/19y908qHph5TIiaGW3AlbuvtvvXxrGbK6/view?usp=share_link "权重文件下载")