# notMNIST_keras
notMNIST implementation with keras and convolutional layers. It uses tensorflow-gpu as a backend. Using a GTX1070, the training takes around 25 minutes and achieves a test accuracy score of around 95%.

The network consists of two convolutional layers followed by a maxpooling layer, followed by four fully connected layers.

## Instructions
1. Run 'data_preprocess.py' to download and format the data
2. Run 'keras_convolution.py'
