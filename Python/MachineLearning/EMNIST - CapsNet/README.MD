This project is an adaptation of the Handson-ML Capsule Network (https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb) implementation.
The original implementation was adapted to work on the extended MNIST dataset.

The capsule network architecture is based off of CapsNet described in "Dynamic routing between capsules"(Sabour et al. 2017). 

That "Dynamic Routing Between Capsules" (Sabour et al. 2017) can be found here:
https://arxiv.org/pdf/1710.09829.pdf

The process of adapting the network involved changing network layer dimensions, optimization function parameters,
and tensorflow training information to account for the fact that the EMNIST dataset used was for 26 alphabetical characters.

Dependencies :

Python 3.X 64-BIT
TensorFlow
Numpy
Scipy

Based on: CapsNet MNIST - Open source Handson-ML Implementation
CapsNet EMNIST Adaptation done by Jesse Broussard
