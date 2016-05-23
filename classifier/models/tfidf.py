import io

from scipy.sparse import csr_matrix

from constants import MODELS_DATA_FOLDER, TF_ONE_SHOT_TRAIN_FILE_PATH, TF_ONE_SHOT_EVAL_FILE_PATH, \
    TF_TFIDF_TRAIN_FILE_PATH, TF_TFIDF_EVAL_FILE_PATH, TF_TFIDF_VECTORIZER_FILE_PATH
from models.utils import variable_with_weight_decay, variable_on_cpu, activation_summary
import tensorflow as tf
import numpy as np


class Tfidf(object):
    TRAIN_DIR = MODELS_DATA_FOLDER + '0.00004_tfidf_train/'
    EVAL_DIR = MODELS_DATA_FOLDER + '0.00004_tfidf_eval/'
    TRAIN_FILE_PATH = TF_TFIDF_TRAIN_FILE_PATH
    EVAL_FILE_PATH = TF_TFIDF_EVAL_FILE_PATH
    VECTORIZER_FILE_PATH = TF_TFIDF_VECTORIZER_FILE_PATH
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 42114
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7395
    NUM_CLASSES = 17
    NUM_VOCABULARY_SIZE = 50701
    TRAIN_RATIO = 0.85
    BATCH_SIZE = 128
    SERIALIZED_FIELD_LEN = 10000

    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 30.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.03  # Initial learning rate.

    FLAGS = tf.app.flags.FLAGS

    @staticmethod
    def inference(images):
        # local1
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(images, [Tfidf.BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value

            weights = variable_with_weight_decay('weights', shape=[dim, 128],
                                                 # stddev=0.01, wd=0.00003)
                                                 # stddev=0.01, wd=0.0001)
                                                 stddev=0.01, wd=0.00004)
            biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
            local1 = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)
            activation_summary(local1)

        # local2
        with tf.variable_scope('local2') as scope:
            weights = variable_with_weight_decay('weights', shape=[128, 64],
                                                 # stddev=0.01, wd=0.00003)
                                                 # stddev=0.01, wd=0.0001)
                                                 stddev=0.01, wd=0.00004)
            biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            local2 = tf.nn.tanh(tf.matmul(local1, weights) + biases, name=scope.name)
            activation_summary(local2)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('logits') as scope:
            if Tfidf.FLAGS.is_training:
                local2 = tf.nn.dropout(local2, 0.8)

            weights = variable_with_weight_decay('weights', [64, Tfidf.NUM_CLASSES],
                                                 stddev=1.0 / Tfidf.BATCH_SIZE, wd=0.0)
            biases = variable_on_cpu('biases', [Tfidf.NUM_CLASSES],
                                     tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local2, weights), biases, name=scope.name)
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
        example = tf.parse_single_example(serialized_example, features={
            'hubs': tf.FixedLenFeature([17], dtype=tf.int64),
            'words': tf.FixedLenFeature([Tfidf.SERIALIZED_FIELD_LEN], dtype=tf.string)
        })

        def unpack_feature(array):
            packed = np.sum(array).strip()

            # print('packed len ', len(packed))
            # to_ubyte = np.vectorize(lambda x: ord(x[0]))
            # byte_array = to_ubyte(array)
            # byte_array -= 32
            # byte_array = np.trim_zeros(byte_array)
            # print('bytes len', len(byte_array))
            # byte_array += 32
            # byte_array = byte_array.astype(np.ubyte)

            input_file = io.BytesIO(packed)
            loader = np.load(input_file)
            matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                shape=loader['shape']).toarray()

            return matrix[0].astype(np.float32)

        label = tf.to_float(example['hubs'])
        label.set_shape((Tfidf.NUM_CLASSES,))

        feature = example['words']
        feature, = tf.py_func(unpack_feature, [feature], [tf.float32])
        feature.set_shape((Tfidf.NUM_VOCABULARY_SIZE,))

        return label, feature
