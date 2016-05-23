# coding=utf-8
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import io
import sys

sys.path.append('/home/dima/univ/techbot/classifier/')

import os
import time
from random import shuffle, random

import ijson
import pickle
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from constants import CATEGORIES, TF_TFIDF_TRAIN_FILE_PATH, TF_TFIDF_EVAL_FILE_PATH, TF_TFIDF_VECTORIZER_FILE_PATH, \
    CATEGORIES_SHORT
from constants import POSTS_FILE_PATH
from models.tfidf import Tfidf


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _get_max_serialized_len(features):
    max_len = 0
    all_len = 0
    for feature in features:
        output = io.BytesIO()
        np.savez_compressed(output, data=feature.data, indices=feature.indices,
                            indptr=feature.indptr, shape=feature.shape)
        feature_str = output.getvalue()
        feature_len = len(feature_str)

        if feature_len > max_len:
            max_len = feature_len
        all_len += feature_len

    elements_num = features.shape[0]
    print('serialized middle len : ', all_len / float(elements_num))
    print('serialized maximum len : ', max_len)
    print('elements : ', elements_num)

    return max_len


def _write_batch(filename, items):
    writer = tf.python_io.TFRecordWriter(filename)
    writed_items = 0
    for feature, label in items:
        feature = feature[0]
        output = io.BytesIO()
        np.savez_compressed(output, data=feature.data, indices=feature.indices,
                            indptr=feature.indptr, shape=feature.shape)

        features_str = output.getvalue()
        if len(features_str) > Tfidf.SERIALIZED_FIELD_LEN:
            continue

        features_str = features_str.rjust(Tfidf.SERIALIZED_FIELD_LEN)
        example = tf.train.Example(features=tf.train.Features(feature={
            'words': _bytes_list_feature(features_str),
            'hubs': _int64_list_feature(label)
        }))
        writer.write(example.SerializeToString())

        writed_items += 1

    return writed_items


def _get_train_eval_indexes():
    train_indexes = set()
    eval_indexes = set()
    with open(POSTS_FILE_PATH, 'r') as f:
        it = 0
        for post_num, post in enumerate(ijson.items(f, 'item')):
            was = False
            for hub in post['hubs']:
                for category_num, category in enumerate(CATEGORIES):
                    if hub in category:
                        was = True

            if not was:
                continue

            if random() > Tfidf.TRAIN_RATIO:
                eval_indexes.add(post_num)
            else:
                train_indexes.add(post_num)

                # it += 1
                # if it > 20:
                #     break

    accepted_posts_num = len(eval_indexes) + len(train_indexes)
    print('eval data set ratio', len(eval_indexes) / float(accepted_posts_num))

    return train_indexes, eval_indexes


def _iterate_features(indexes):
    with open(POSTS_FILE_PATH, 'r') as f:
        it = 0
        for post_num, post in enumerate(ijson.items(f, 'item')):
            # it += 1
            # if it > 30:
            #     break

            if post_num in indexes:
                post_words = post['content'] + post['title']
                yield ' '.join(post_words)


def _get_labels(train_indexes, eval_indexes):
    train_labels = []
    eval_labels = []
    it = 0
    with open(POSTS_FILE_PATH, 'r') as f:
        for post_num, post in enumerate(ijson.items(f, 'item')):
            # it += 1
            # if it > 30:
            #     break

            is_in_train_set = post_num in train_indexes
            is_in_eval_set = post_num in eval_indexes
            if not is_in_train_set and not is_in_eval_set:
                continue

            label = [0] * len(CATEGORIES)
            for hub in post['hubs']:
                for category_num, category in enumerate(CATEGORIES):
                    if hub in category:
                        label[category_num] = 1

            if is_in_train_set:
                train_labels.append(label)
            else:
                eval_labels.append(label)

    return train_labels, eval_labels


def _get_batch(features, labels):
    batch = list(zip(features, labels))
    shuffle(batch)

    return batch


def write_data():
    train_indexes, eval_indexes = _get_train_eval_indexes()

    vectorizer = TfidfVectorizer(min_df=7)
    train_labels, eval_labels = _get_labels(train_indexes, eval_indexes)

    train_features = vectorizer.fit_transform(_iterate_features(train_indexes))
    eval_features = vectorizer.transform(_iterate_features(eval_indexes))

    with open(Tfidf.VECTORIZER_FILE_PATH, 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_max_len = _get_max_serialized_len(train_features)
    eval_max_len = _get_max_serialized_len(eval_features)
    # serialized_max_len = max(train_max_len, eval_max_len)

    train_batch = _get_batch(train_features, train_labels)
    eval_batch = _get_batch(eval_features, eval_labels)

    train_set_size = _write_batch(Tfidf.TRAIN_FILE_PATH, train_batch)
    eval_set_size = _write_batch(Tfidf.EVAL_FILE_PATH, eval_batch)
    vocabulary_size = len(vectorizer.vocabulary_)

    set_size = train_set_size + eval_set_size
    print('Set size : ', set_size)
    print('Train set size : ', train_set_size)
    print('Eval set size : ', eval_set_size)
    print('Vocabulary size : ', vocabulary_size)

    # Tfidf.NUM_VOCABULARY_SIZE = vocabulary_size

    # clf = Ridge(alpha=1.0)
    # clf.fit(train_features, train_labels)
    # accuracy = clf.score(eval_features, eval_labels)
    # print('Accuracy : ', accuracy)


def main():
    if os.path.exists(TF_TFIDF_TRAIN_FILE_PATH) or \
            os.path.exists(TF_TFIDF_EVAL_FILE_PATH):
        print('TF records already exists')
        return

    print('TF data writing started')
    start_time = time.time()
    write_data()
    format_str = 'TF data writing finished in {} seconds'
    print(format_str.format(time.time() - start_time))


def test_input():
    filename_queue = tf.train.string_input_producer([Tfidf.EVAL_FILE_PATH])

    labels, features = Tfidf.read_news(filename_queue)

    sess = tf.Session()

    # Required. See below for explanation
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    with open(Tfidf.VECTORIZER_FILE_PATH, 'rb') as handle:
        vectorizer = pickle.load(handle)

    for i in range(10000):
        print(i)
        label_val, feature_val = sess.run([labels, features])

        # assert len(label_val) == Tfidf.NUM_CLASSES
        # assert len(feature_val) == Tfidf.NUM_VOCABULARY_SIZE

        def print_unicode(items):
            print('array')
            res = ''
            for string in items:
                res += string + ' '
            print(res)

            # print_unicode(np.array(CATEGORIES_SHORT)[np.nonzero(label_val)[0]])
            # print_unicode(np.array(vectorizer.get_feature_names())[np.nonzero(feature_val)[0]])
            # print('\n\n\n')


def test_random():
    eval_size = 0
    for _ in range(40000):
        num = random()
        if num > 0.85:
            eval_size += 1
    print(eval_size / 40000.0)


if __name__ == '__main__':
    main()
    # test_random()
    # test_input()
