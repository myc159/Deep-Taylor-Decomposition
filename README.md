# Deep Taylor Decomposition

### This implementation codes are specifically designed to see the visual result of deep taylor decomposition, a novel saliency mapping methods for deep neural network, applied at ImageNet pretrained models such as vgg or resnet.

**Abstract** - Nonlinear methods such as Deep Neural Networks (DNNs) are the gold standard for various challenging machine learning problems, e.g., image classification, natural language processing or human action recognition. Although these methods perform impressively well, they have a significant disadvantage, the lack of transparency, limiting the interpretability of the solution and thus the scope of application in practice. Especially DNNs act as black boxes due to their multilayer nonlinear structure. In this paper we introduce a novel methodology for interpreting generic multilayer neural networks by decomposing the network classification decision into contributions of its input elements. Although our focus is on image classification, the method is applicable to a broad set of input data, learning tasks and network architectures. Our method is based on deep Taylor decomposition and efficiently utilizes the structure of the network by backpropagating the explanations from the output to the input layer. We evaluate the proposed method empirically on the MNIST and ILSVRC data sets. - https://arxiv.org/abs/1512.02479

## Results

### vgg16bn-ImageNet pretrained

| ![origin1](./sample_image/origin1.png) | ![vgg16bn_1](./sample_image/vgg16bn_1.png) | ![origin2](./sample_image/origin2.png) | ![vgg16bn_2](./sample_image/vgg16bn_2.png) | ![origin3](./sample_image/origin3.png) | ![vgg16bn_3](./sample_image/vgg16bn_3.png) |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![origin4](./sample_image/origin4.png) | ![vgg16bn_4](./sample_image/vgg16bn_4.png) | ![origin5](./sample_image/origin5.png) | ![vgg16bn_5](./sample_image/vgg16bn_5.png) | ![origin6](./sample_image/origin6.png) | ![vgg16bn_6](./sample_image/vgg16bn_6.png) |

### vgg19-ImageNet pretrained

| ![origin1](./sample_image/origin1.png) | ![vgg19_1](./sample_image/vgg19_1.png) | ![origin2](./sample_image/origin2.png) | ![vgg19_2](./sample_image/vgg19_2.png) | ![origin3](./sample_image/origin3.png) | ![vgg19_3](./sample_image/vgg19_3.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![origin4](./sample_image/origin4.png) | ![vgg19_4](./sample_image/vgg19_4.png) | ![origin5](./sample_image/origin5.png) | ![vgg19_5](./sample_image/vgg19_5.png) | ![origin6](./sample_image/origin6.png) | ![vgg19_6](./sample_image/vgg19_6.png) |

### resnet34-ImageNet pretrained

| ![origin1](./sample_image/origin1.png) | ![res34_1](./sample_image/res34_1.png) | ![origin2](./sample_image/origin2.png) | ![res34_2](./sample_image/res34_2.png) | ![origin3](./sample_image/origin3.png) | ![res34_3](./sample_image/res34_3.png) |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![origin4](./sample_image/origin4.png) | ![res34_4](./sample_image/res34_4.png) | ![origin5](./sample_image/origin5.png) | ![res34_5](./sample_image/res34_5.png) | ![origin6](./sample_image/origin6.png) | ![res34_6](./sample_image/res34_6.png) |

### resnet101-ImageNet pretrained

| ![origin1](./sample_image/origin1.png) | ![res101_1](./sample_image/res101_1.png) | ![origin2](./sample_image/origin2.png) | ![res101_2](./sample_image/res101_2.png) | ![origin3](./sample_image/origin3.png) | ![res101_3](./sample_image/res101_3.png) |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![origin4](./sample_image/origin4.png) | ![res101_4](./sample_image/res101_4.png) | ![origin5](./sample_image/origin5.png) | ![res101_5](./sample_image/res101_5.png) | ![origin6](./sample_image/origin6.png) | ![res101_6](./sample_image/res101_6.png) |

## Future Work

* Densenet pretrained model version will be updated soon.
* Batch normalization layer need some modification.
