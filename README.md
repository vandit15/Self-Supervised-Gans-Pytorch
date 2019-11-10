# Self-Supervised-Gans-Pytorch
Pytorch implementation of the CVPR'19 paper "Self-Supervised GANs via Auxiliary Rotation Loss"


Ting Chen, [Xiaohua Zhai](xzhai@google.com)(Google Brain), [Marvin Ritter](marvinritter@google.com)(Google Brain), [Mario Lucic](lucic@google.com)(Google Brain), [Neil Houlsby](neilhoulsby@google.com)(Google Brain)


## Dependencies
- Python (>=3.6)
- Pytorch (>=1.2.0) 

## Training
`python main.py`

## How it works
The paper presents a method to combine adverserial training with self-supervised learning. It uses the concept of Auxilliary Rotation Loss. The main idea behind self-supervision is to train a model on a pretext task like predicting rotation angle and then extracting representations from the resulting networks. The discriminator also tries to predict the rotatory angle(0, 90, 180, 270) along with the normal prediction of fake vs real.

## References
[official tensorflow implementation](https://github.com/google/compare_gan/tree/master/compare_gan)
[unofficial tensorflow implementation](https://github.com/zhangqianhui/Self-Supervised-GANs)
