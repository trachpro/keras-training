import keras
<<<<<<< HEAD
from nets.lenet.lenet5 import LeNet5, TuNet
=======
from nets.lenet.lenet5 import LeNet5, thanh_net, TrongNet
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
from keras.optimizers import Adadelta
import os
import datetime
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from keras import backend as K


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#datagen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator()
<<<<<<< HEAD
BATCH_SIZE = 128 * 2
N_CLASSES = 38
=======
BATCH_SIZE = 32
N_CLASSES = 37
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
#LR = 0.001
N_EPOCHS = 40
IMG_SIZE = 50

<<<<<<< HEAD
train_path ='/home/ai-team/projects/tupm/character/out'
val_path='/home/ai-team/projects/tupm/character/val'
=======
train_path ='/home/tupm/Projects/text-gen-ocr/out'
val_path='/home/tupm/Projects/text-gen-ocr/val'
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4

# train data, label
# datagen.fr
train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode ='sparse', color_mode="grayscale")

<<<<<<< HEAD
f = open('/home/ai-team/projects/tupm/character/label_out.txt', "w+")
=======
f = open('/home/tupm/Projects/text-gen-ocr/label_out.txt', "w+")
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
class_lable = train_generator.class_indices
class_lable = list(class_lable.keys())
for i in range(len(class_lable)):
        f.write(class_lable[i])
f.close()
# val data, label
validation_generator = datagen.flow_from_directory(
        val_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode ='sparse',color_mode="grayscale")

#load old model with weight
model = LeNet5(input_shape=(50,50,1), num_classes=N_CLASSES)
#model.load_weights('/home/thanh/Downloads/keras-training/date_model/201904251621/weights-improvement-40-0.97-0.50-0.11.hdf5')
model.summary()
<<<<<<< HEAD
save_model_path = os.path.join('/home/ai-team/HDD/tupm/keras-training/pass-model','{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M')))
=======
save_model_path = os.path.join('/home/tupm/Projects/keras-training/pass-model','{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M')))
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
os.makedirs(save_model_path, exist_ok= True)
model_file = os.path.join(save_model_path, "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5")
ModelCheckpoint = keras.callbacks.ModelCheckpoint
checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/tensorboard', write_graph=True, write_images=True)
# #
callbacks_list = [checkpoint, tbCallBack]

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(lr=0.001),
              metrics=['accuracy'])
model.fit_generator(
        train_generator,
<<<<<<< HEAD
        steps_per_epoch= 175560 // BATCH_SIZE,
=======
        steps_per_epoch= 10656 // BATCH_SIZE,
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
        initial_epoch=0,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps= 5920 // BATCH_SIZE,
        callbacks=callbacks_list,
<<<<<<< HEAD
        max_queue_size=12,
        workers=6,
=======
        max_queue_size=4,
        workers=2,
>>>>>>> 2e246e98271f3b7290db8d63c93da1bc1fd00cf4
        use_multiprocessing=True,
        )
model.save(os.path.join(save_model_path,"date_model.hdf5"))
print('model is saved')

output_model = 'date_model.pb'

converted_output_node_names = [node.op.name for node in model.outputs]

sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), converted_output_node_names)
graph_io.write_graph(constant_graph, str('pb_model'), output_model, as_text=False)
