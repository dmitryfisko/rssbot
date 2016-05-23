from constants import MODELS_DATA_FOLDER, TF_ONE_SHOT_TRAIN_FILE_PATH, TF_ONE_SHOT_EVAL_FILE_PATH
from models.utils import variable_with_weight_decay, variable_on_cpu, activation_summary
import tensorflow as tf
import numpy as np


class BagOfWords(object):
    TRAIN_DIR = MODELS_DATA_FOLDER + 'one_hot_train/'
    EVAL_DIR = MODELS_DATA_FOLDER + 'one_hot_eval/'
    TRAIN_FILE_PATH = TF_ONE_SHOT_TRAIN_FILE_PATH
    EVAL_FILE_PATH = TF_ONE_SHOT_EVAL_FILE_PATH
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 42731
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7541
    NUM_CLASSES = 17
    NUM_VOCABULARY_SIZE = 50000
    TRAIN_RATIO = 0.85
    BATCH_SIZE = 128

    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.03  # Initial learning rate.

    @staticmethod
    def inference(images):
        # local
        with tf.variable_scope('local') as scope:
            reshape = tf.reshape(images, [BagOfWords.BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value

            weights = variable_with_weight_decay('weights', shape=[dim, BagOfWords.NUM_CLASSES],
                                                 stddev=0.04, wd=0.0001)
            biases = variable_on_cpu('biases', [BagOfWords.NUM_CLASSES], tf.constant_initializer(0.1))
            local = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)
            activation_summary(local)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            weights = variable_with_weight_decay('weights', [BagOfWords.NUM_CLASSES, BagOfWords.NUM_CLASSES],
                                                 stddev=1.0 / BagOfWords.BATCH_SIZE, wd=0.0)
            biases = variable_on_cpu('biases', [BagOfWords.NUM_CLASSES],
                                     tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local, weights), biases, name=scope.name)
            activation_summary(softmax_linear)

        return softmax_linear

    @staticmethod
    def read_news(filename_queue):
        """Reads and parses examples from CIFAR10 data files.

        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.

        Args:
        filename_queue: A queue of strings with the filenames to read from.
        """

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'hubs': tf.FixedLenFeature([3], dtype=tf.int64),
            'words': tf.FixedLenFeature([6250], dtype=tf.int64)
        })

        def unpackbits(arr):
            arr = arr.astype(np.ubyte)
            unpacked_arr = np.unpackbits(arr)
            if len(unpacked_arr) == 24:
                unpacked_arr = unpacked_arr[:BagOfWords.NUM_CLASSES]
            return unpacked_arr.astype(np.float32)

        labels = features['hubs']
        labels, = tf.py_func(unpackbits, [labels], [tf.float32])
        labels.set_shape((BagOfWords.NUM_CLASSES,))

        words = features['words']
        words, = tf.py_func(unpackbits, [words], [tf.float32])
        words.set_shape((BagOfWords.NUM_VOCABULARY_SIZE,))

        return labels, words
