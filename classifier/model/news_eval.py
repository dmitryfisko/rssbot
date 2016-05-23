# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
news_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by news_eval.py.

Speed:
On a single Tesla K40, news_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import os

import numpy as np
import tensorflow as tf

sys.path.insert(0, "/home/dima/univ/techbot/classifier/")
sys.path.insert(1, "/home/dima/univ/techbot/classifier/model/")
os.chdir('../../')

from model import news
from model.interface import Model

RUN_ONCE = False
EVAL_INTERVAL_SECS = 60 * 60
IS_EVAL_DATA = False

tf.app.flags.DEFINE_string('checkpoint_dir', Model.TRAIN_DIR,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('is_training', False,
                           """Directory where to read model checkpoints.""")

if IS_EVAL_DATA:
    NUM_EXAMPLES = Model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
else:
    NUM_EXAMPLES = Model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


def eval_once(saver, summary_writer, logits_ind, labels, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(Model.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            print('Model checkpoint path', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # threads = []
            # for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            #     threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
            #                                      start=True))

            num_iter = int(math.ceil(NUM_EXAMPLES / Model.BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * Model.BATCH_SIZE
            step = 0
            first_logit = None
            while step < num_iter and not coord.should_stop():
                logits_ind_val, labels_val = sess.run([logits_ind, labels])
                logits_ind_val = logits_ind_val.flatten()
                if step == 0:
                    first_logit = logits_ind_val
                for ind in range(len(labels_val)):
                    if logits_ind_val[ind] in np.nonzero(labels_val[ind])[0]:
                        true_count += 1
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            precision = None
            coord.request_stop(e)

        coord.request_stop()
        first_logit_str = ' '.join([str(item).rjust(3, ' ') for item in first_logit])
        return precision, first_logit_str


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        global IS_EVAL_DATA
        IS_EVAL_DATA = IS_EVAL_DATA
        images, labels = news.inputs(is_eval_data=IS_EVAL_DATA)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = Model.inference(images)

        # labels_num = tf.reduce_sum(tf.to_int32(labels), 1)
        _, logits_ind = tf.nn.top_k(logits, 1)
        # _, logits_ind = tf.nn.top_k(logits, labels_num)
        # predictions = tf.equal(labels_ind, logits_ind)

        # res = labels_ind == logits_ind
        # Calculate predictions.
        # labels = tf.reshape(tf.concat(1, labels), [-1, labels_num])
        # top_k_op = tf.nn.in_top_k(logits, labels_ind, 1)

        # tf.add(logits, 1)
        # largest_index = tf.argmax(logits, 1)
        # top_k_op =  labels[largest_index]:
        #     top_k_op = tf.constant(1, dtype=tf.int32)
        # else:
        #     top_k_op = tf.constant(0, dtype=tf.int32)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            Model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(Model.EVAL_DIR, g)

        prefix = 'eval' if IS_EVAL_DATA else 'train'
        while True:
            score, logit = eval_once(saver, summary_writer, logits_ind, labels, summary_op)
            str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            with open(Model.EVAL_DIR + prefix + '_accuracy.txt', 'a') as f:
                f.write('{} {} score : {}\n'.format(str_time, prefix, score))
                f.write('{}\n\n'.format(logit))

            if RUN_ONCE:
                break
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
