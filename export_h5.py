from tensorflow.python import keras
from nets.lenet.lenet5 import LeNet5, TrongNet
from keras.applications.vgg16 import VGG16

model = LeNet5((50, 50, 1), 37, "/home/tupm/Projects/keras-training/pass-model/weights-improvement-40-0.55.hdf5")
model.save("/home/tupm/Projects/keras-training/h5-model/weights-improvement-40-0.55.h5")
