


"""
Implementation of example defense.
This defense loads inception v1 checkpoint and classifies all images using loaded checkpoint.
"""

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import vgg

from scipy.misc import imread
from scipy.misc import imresize
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import Model
from PIL import Image
slim = tf.contrib.slim
'''
inception_v1/inception_v1.ckpt
vgg_16/vgg_16.ckpt
resnet_v1_50/model.ckpt-49800

'''

#写文件夹名称
module_file =  tf.train.latest_checkpoint('../models/inception_v1')
print('module_file',module_file)


tf.flags.DEFINE_string(
    'checkpoint_path', module_file, 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', 'F:\\陶士来文件\\tsl_python_project\\model_datas\\tianchi_ijcai\\IJCAI_2019_AAAC_dev_data\\dev_data', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '../output_res', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
#inception时可以16,vgg
tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with open(os.path.join(input_dir, 'dev.csv'), 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['filename'])
            with open(filepath) as f:
                raw_image = imread(filepath, mode='RGB').astype(np.float)
                # image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0  # for inception
                image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width])  # for vgg and resnet v1
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            # images[idx, :, :, :] = image * 2.0 - 1.0  # for inception
            image = image - np.array([[123.68, 116.78, 103.91]]).reshape((1, 1, 3))  # for vgg and resnet v1
            images[idx, :, :, :] = image / 160.0
            labels[idx] = int(row['targetedLabel'])
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros(batch_shape[0], dtype=np.int32)
                gt_labels = np.zeros(batch_shape[0], dtype=np.int32)
                idx = 0

        if idx > 0:
            yield filenames, images, labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with open(os.path.join(output_dir, filename), 'w') as f:
            filepath = os.path.join(output_dir, filename)
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            # resize back to [299, 299]
            r_img = imresize(img, [299, 299])
            Image.fromarray(r_img).save(filepath, format='PNG')


class InceptionModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            _, end_points = inception.inception_v1(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = end_points['Logits']
        # Strip off the extra reshape op at the output
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)

class ResNetModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(ResNetModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, end_points = resnet_v1.resnet_v1_50(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = tf.squeeze(end_points['resnet_v1_50/logits'])
        # Strip off the extra reshape op at the output
        self.probs = end_points['predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)

class VGGModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(VGGModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(
                x_input, num_classes=self.nb_classes, is_training=False)
        self.built = True
        self.logits = tf.squeeze(end_points['vgg_16/fc8'])
        # Strip off the extra reshape op at the output
        self.probs = tf.nn.softmax(end_points['vgg_16/fc8'])
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)


def main(_):
    """Run the sample attack"""
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes
    tf.logging.set_verbosity(tf.logging.INFO)


    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class = tf.one_hot(target_class_input, nb_classes)
        model = InceptionModel(nb_classes)
        # model = ResNetModel(nb_classes)
        # model = VGGModel(nb_classes)

        # Run computation

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu90%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        with tf.Session(config=config) as sess:
            mim = MomentumIterativeMethod(model, sess=sess)
            attack_params = {"eps": 32.0 / 255.0, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 20, "decay_factor": 1.0, "y_target": one_hot_target_class}
            x_adv = mim.generate(x_input, **attack_params)
            saver = tf.train.Saver(slim.get_model_variables())
            saver.restore(sess, FLAGS.checkpoint_path)
            for filenames, images, tlabels in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv,
                                      feed_dict={x_input: images, target_class_input: tlabels})

                save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
    # pass
    tf.app.run()