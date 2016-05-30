from collections import Counter
from random import random

import ijson
import operator
from sklearn.feature_extraction import DictVectorizer

from classifier.constants import POSTS_FILE_PATH, CATEGORIES, XGBOOST_PICKLE_CLF, XGBOOST_PICKLE_VECTORIZER
import numpy as np
import pickle

from classifier.models.xgb_classifier import XgbClassifier

TRAIN_RATIO = 0.85

DEBUG = False


def _get_dataset_indexes():
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

            if random() > TRAIN_RATIO:
                eval_indexes.add(post_num)
            else:
                train_indexes.add(post_num)

            if DEBUG:
                it += 1
                if it > 100:
                    break

    accepted_posts_num = len(eval_indexes) + len(train_indexes)
    print('eval data set ratio', len(eval_indexes) / float(accepted_posts_num))
    return train_indexes, eval_indexes


def _iterate_features(indexes):
    with open(POSTS_FILE_PATH, 'r') as f:
        it = 0
        for post_num, post in enumerate(ijson.items(f, 'item')):
            if DEBUG:
                it += 1
                if it > 100:
                    break

            if post_num in indexes:
                post_words = post['content'] + post['title']
                yield Counter(post_words)


def _get_labels(train_indexes, eval_indexes):
    train_labels = []
    eval_labels = []
    it = 0
    with open(POSTS_FILE_PATH, 'r') as f:
        for post_num, post in enumerate(ijson.items(f, 'item')):
            if DEBUG:
                it += 1
                if it > 100:
                    break

            is_in_train_set = post_num in train_indexes
            is_in_eval_set = post_num in eval_indexes
            if not is_in_train_set and not is_in_eval_set:
                continue

            label = [0.] * len(CATEGORIES)
            for hub in post['hubs']:
                for category_num, category in enumerate(CATEGORIES):
                    if hub in category:
                        label[category_num] = 1

            if is_in_train_set:
                train_labels.append(label)
            else:
                eval_labels.append(label)

    return train_labels, eval_labels


def _process_train_labels(labels):
    counter = np.array([0] * len(CATEGORIES))
    categories = []
    for label in labels:
        categories_indexes = np.nonzero(label)
        frequencies = counter[categories_indexes]
        _, category_index = min(zip(frequencies, *categories_indexes),
                                key=operator.itemgetter(0))
        categories.append(category_index)
        counter[category_index] += 1

    return categories


def _get_score(preds, labels):
    true_count = 0
    for pred, label in zip(preds, labels):
        if label[int(pred)]:
            true_count += 1

    return float(true_count) / len(labels)


def main():
    train_indexes, eval_indexes = _get_dataset_indexes()

    vectorizer = DictVectorizer()
    train_labels, eval_labels = _get_labels(train_indexes, eval_indexes)
    train_labels = _process_train_labels(train_labels)

    train_features = vectorizer.fit_transform(_iterate_features(train_indexes))
    eval_features = vectorizer.transform(_iterate_features(eval_indexes))

    clf = XgbClassifier(eta=0.09, min_child_weight=6, depth=20, num_round=150, threads=4)
    clf.train(train_features, train_labels)
    eval_predicts = clf.predict(eval_features)

    score = _get_score(eval_predicts, eval_labels)
    print('accuracy {}'.format(score))
    print(eval_predicts.tolist())

    with open(XGBOOST_PICKLE_CLF, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(XGBOOST_PICKLE_VECTORIZER, 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
