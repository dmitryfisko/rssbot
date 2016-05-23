# coding=utf-8
from __future__ import print_function
from __future__ import print_function

import json
import os
import time
from collections import defaultdict, Counter

import ijson

from constants import WORDS_FILE_PATH, POSTS_FILE_PATH

SHOW_PROGRESS_EVERY = 100000


def add_words(words, text, priority):
    for word in text:
        words[word] += priority


def save_words(words):
    counter = Counter(words)
    with open(WORDS_FILE_PATH, 'w') as outfile:
        json.dump(counter.most_common(), outfile)


def show_words():
    with open(WORDS_FILE_PATH, 'r') as infile:
        words = json.load(infile)

        print('size ', len(words))
        for k, v in words[:50000]:
            print(u'{} : {}'.format(k, v))


def extract_words():
    words = defaultdict(int)
    iteration = 0
    with open(POSTS_FILE_PATH, 'r') as f:
        for post in ijson.items(f, 'item'):
            add_words(words, post['content'], 1)
            add_words(words, post['title'], 2)
            add_words(words, post['tags'], 3)

        iteration += 1
        if not iteration % SHOW_PROGRESS_EVERY:
            format_str = 'Words parsing {} iterations passed'
            print(format_str.format(iteration))

    return words


def main():
    if os.path.exists(WORDS_FILE_PATH):
        print('Words data already exist')
        return

    print('Words parsing started')
    start_time = time.time()
    words = extract_words()
    save_words(words)
    format_str = 'Words parsing finished in {} seconds'
    print(format_str.format(time.time() - start_time))


if __name__ == '__main__':
    # main()
    show_words()
