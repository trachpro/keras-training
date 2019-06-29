from tensorflow.python import keras
from nets.lenet.lenet5 import LeNet5, TrongNet
from keras.applications.vgg16 import VGG16

model = TrongNet((50, 50, 1), 10, "/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904160859_lenet5_12/weights-181-0.99.hdf5")
model.save("/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904160859_lenet5_12/weights-181-0.99.h5")
