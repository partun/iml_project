from tensorflow.keras.optimizers import RMSprop
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from ImageDataGeneratorCustom import ImageDataGeneratorCustom
from helper import gen_path_files, prediction_generator

'''
    Hostettler Maurice, Dominic Steiner

    packages:
    - numpy
    - tensorflow
    - keras
    - sklearn
    - scipy
    - skimage install with $pip install scikit-image
    - tqdm install with $pip install tqdm

    the 10000 images should be named 00000.jpg to 09999.jpg and be in folder ./data/food/
    
    - training triplets stored in ./train_triplets.txt
    - testing triplets stored in ./train_triplets.txt

    - predictions will be stored in ./output.txt

    - weights of the retrained inception model stored in file ./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
      the weights can be downloaded from: https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

'''


parser = argparse.ArgumentParser()
parser.add_argument('-loadweights', dest='loadweights',
                    action='store_true')  # load save weights
parser.add_argument('-notrain', dest='notrain',
                    action='store_true')   # skip training
parser.add_argument('-nopredict', dest='nopredict',
                    action='store_true')  # skip prediction step
args = parser.parse_args()
print(args)

# fixing seeds
seed_value = 46
tf.random.set_seed(seed_value)


def triplt_loss(y_true, y_pred):
    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    total_loss = tf.convert_to_tensor(0, dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i]
            p_embedding = y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p = K.sqrt(K.sum((q_embedding-p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding-n_embedding)**2))
            loss = tf.maximum(g+D_q_p-D_q_n, zero)
            total_loss = total_loss+loss
        except:
            continue
    total_loss = total_loss/(batch_size/3)
    return total_loss


def _metric_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i+0]
            p_embedding = y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p = K.sqrt(K.sum((q_embedding - p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
            loss = loss + tf.cond(K.less(D_q_p, D_q_n), lambda: tf.constant(1.0, shape=[
                                  1], dtype=tf.float32), lambda: tf.constant(0.0, shape=[1], dtype=tf.float32))
        except:
            continue

    loss = loss / (batch_size / 3)
    return loss


class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory("./data/",
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=False,
                                            triplet_path='./path_train_triplets.txt'
                                            )

    def get_val_generator(self, batch_size):
        return self.idg.flow_from_directory("./data/",
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=False,
                                            triplet_path='./path_val_triplets.txt'
                                            )


dg = DataGenerator({
    "rescale": 1. / 255,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.3,
    "shear_range": 0.3,
    "rotation_range": 45,
    "fill_mode": 'nearest',
    "data_format": 'channels_last'
    # "samplewise_center": True
}, target_size=(224, 224))


batch_size, train_triplets_count, val_triplets_count = gen_path_files(
    'train_triplets.txt', './data/food/', p=0.1)
batch_size *= 3
train_generator = dg.get_train_generator(batch_size)
val_generator = dg.get_val_generator(batch_size)

_EPSILON = K.epsilon()

image_shape = (224, 224, 3)
model_path = 'inception_model.h5'
pretrained_model = InceptionV3(
    input_shape=image_shape, include_top=False, weights=None)
pretrained_model.load_weights(
    './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

for layer in pretrained_model.layers:
    if layer.name == 'mixed6':
        layer.trainable = True
    else:
        layer.trainable = False

last_layer = pretrained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dropout(0.6)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(2048)(x)


model = Model(pretrained_model.input, x)

if args.loadweights:
    model.load_weights(model_path)
    print('loaded trained model weights.')

model.compile(loss=triplt_loss,
              optimizer=RMSprop(lr=0.0001),
              metrics=[_metric_tensor])

train_steps_per_epoch = train_triplets_count * 3 / batch_size
validation_steps = val_triplets_count * 3 / batch_size
train_epochs = 7

if not args.notrain:
    model.fit(train_generator,  
            steps_per_epoch=train_steps_per_epoch,
            validation_data=val_generator,
            validation_batch_size=batch_size,
            validation_steps=validation_steps,
            epochs=train_epochs,
            callbacks=[],
            initial_epoch=0)
    model.save_weights(model_path)

if not args.nopredict:
    print('starting calculating preditions...')
    pg = prediction_generator('./data/food/', model, nm=True)
    pg.generate_output('test_triples.txt')
    print('Done! ... preditions are stored in output.txt')
