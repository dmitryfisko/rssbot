# coding=utf-8
import json
import os
from random import shuffle

import tensorflow as tf
import numpy as np

import ijson
import time

from constants import WORDS_FILE_PATH, HUBS_FILE_PATH, TF_ONE_SHOT_EVAL_FILE_PATH, CATEGORIES

from constants import POSTS_FILE_PATH, TF_ONE_SHOT_TRAIN_FILE_PATH
from models.bagofwords import BagOfWords

HUBS_MIN_MENTIONS = 50


def _get_dictionary_words():
    words = dict()
    with open(WORDS_FILE_PATH, 'r') as words_file:
        words_all = json.load(words_file)
        for i, (word, _) in enumerate(words_all[:BagOfWords.NUM_VOCABULARY_SIZE - 1]):
            words[word] = i

    return words


def _get_hubs():
    hubs = dict()
    with open(HUBS_FILE_PATH, 'r') as hubs_file:
        hubs_all = json.load(hubs_file)
        for i, (word, mentions) in enumerate(hubs_all):
            if mentions >= HUBS_MIN_MENTIONS:
                hubs[word] = i

    return hubs


def _bytes_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _write_set(filename, sparse_items):
    writer = tf.python_io.TFRecordWriter(filename)
    for hubs_ind, words_ind in sparse_items:
        example = tf.train.Example(features=tf.train.Features(feature={
            'words': _bytes_list_feature(words_ind),
            'hubs': _bytes_list_feature(hubs_ind)
        }))
        writer.write(example.SerializeToString())


def write_data(dict_words):
    items = []
    with open(POSTS_FILE_PATH, 'r') as f:
        for post in ijson.items(f, 'item'):
            labels = [0] * len(CATEGORIES)
            was = False
            for hub in post['hubs']:
                for cur_label, label in enumerate(CATEGORIES):
                    if hub in label:
                        labels[cur_label] = 1
                        was = True

            if not was:
                continue

            words = [0] * BagOfWords.NUM_VOCABULARY_SIZE
            post_words = post['content'] + post['title']
            for word in post_words:
                if word in dict_words:
                    words[dict_words[word]] = 1
                else:
                    words[BagOfWords.NUM_VOCABULARY_SIZE - 1] = 1

            labels = np.packbits(labels).tolist()
            words = np.packbits(words).tolist()
            items.append((labels, words))

    shuffle(items)
    train_set_size = int(len(items) * BagOfWords.TRAIN_RATIO)
    _write_set(TF_ONE_SHOT_TRAIN_FILE_PATH, items[:train_set_size])
    _write_set(TF_ONE_SHOT_EVAL_FILE_PATH, items[train_set_size:])

    print('Set size : ', len(items))
    print('Train set size : ', train_set_size)
    print('Eval set size : ', len(items) - train_set_size)


def main():
    if os.path.exists(TF_ONE_SHOT_TRAIN_FILE_PATH) or \
            os.path.exists(TF_ONE_SHOT_EVAL_FILE_PATH):
        print('TF records already exists')
        return

    print('TF data writing started')
    start_time = time.time()
    dict_words = _get_dictionary_words()
    # max_posts_num = _get_posts_num()
    # print('posts num : ', max_posts_num)
    write_data(dict_words)
    format_str = 'TF data writing finished in {} seconds'
    print(format_str.format(time.time() - start_time))


def test_input():
    filename_queue = tf.train.string_input_producer([TF_ONE_SHOT_EVAL_FILE_PATH])

    labels, words = BagOfWords.read_news(filename_queue)

    sess = tf.Session()

    # Required. See below for explanation
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for i in range(50000):
        labels_val, words_val = sess.run([labels, words])
        assert len(labels_val) == 17
        assert len(words_val) == 50000


if __name__ == '__main__':
    # main()
    test_input()
