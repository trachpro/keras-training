from tensorflow.python import keras
from nets.lenet.lenet5 import LeNet5, TrongNet, TuNet
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from keras import backend as K

model = LeNet5((50, 50, 1), 38, "/media/ai-team/HDD/tupm/keras-training/pass-model/201908011537/date_model.hdf5")

output_model = 'date_model1.pb'

converted_output_node_names = [node.op.name for node in model.outputs]

sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), converted_output_node_names)
graph_io.write_graph(constant_graph, str('pb_model'), output_model, as_text=False)