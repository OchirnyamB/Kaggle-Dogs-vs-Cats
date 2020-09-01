# Kaggle-Dogs-vs-Cats
Writing an algorithm to classify whether images contain either a dog or a cat.

### Requirements:
* python3
* keras
* tensorflow
* numpy V1.19.2
* scikit-image V0.17.2
* opencv-python
* matplotlib V3.2.2
* imutils 

### CNN architecture implemented in this repository:
3. **AlexNet** network structure table summary:

Layer Type | Output Size | Filter Size / Stride
---------- | ------------| -----------
INPUT IMAGE | 227x227x3 |
CONV | 55x55x96 | 11x11/4x4, K=96
ACT  | 55x55x96 | 
BN | 55x55x96 | 
POOL | 27X27X96 | 3x3/2x2
DROPOUT | 27x27x96 |
CONV | 27x27x256 | 5x5, K=256
ACT  | 27x27x256 | 
BN | 27x27x256 | 
POOL | 13X13X256 | 3x3/2x2
DROPOUT | 13x13x256 |
CONV | 13x13x384 | 3x3, K=384
ACT  | 13x13x384 | 
BN | 13x13x384 | 
CONV | 13x13x384 | 3x3, K=384
ACT  | 13x13x384 | 
BN | 13x13x384 | 
CONV | 13x13x256 | 3x3, K=256
ACT  | 13x13x256 | 
BN | 13x13x256 | 
POOL | 6X6X256 | 3x3/2x2
DROPOUT | 6x6x256|
FC | 4096 |
ACT | 4096 |
BN | 4096 |
DROPOUT |4096|
FC | 4096 |
ACT | 4096 |
BN | 4096 |
DROPOUT | 4096 |
FC | 1000 |
SOFTMAX | 1000 |

### Kaggle Dogs vs Cats Dataset:
[The dataset contains 25,000 images of dogs and cats](https://www.kaggle.com/c/dogs-vs-cats/data)

### Evaluations of the Trained Networks:


### References:
* Deep Learning for Computer Vision with Python VOL1 & VOL2 by Dr.Adrian Rosebrock



