# coding=utf-8
from __future__ import print_function

import json
import os
import sys
import time
from collections import defaultdict, Counter

from constants import HUBS_FILE_PATH, POSTS_FILE_PATH

import ijson

SHOW_PROGRESS_EVERY = 10000


def save_hubs(hubs):
    counter = Counter(hubs)
    with open(HUBS_FILE_PATH, 'w') as outfile:
        json.dump(counter.most_common(), outfile)


def show_hubs_info():
    with open(HUBS_FILE_PATH, 'r') as infile:
        hubs = json.load(infile)

        hubs_count = 0
        print('Hubs count {}'.format(len(hubs)))
        for name, mentions in hubs:
            hubs_count += mentions
            print(u'{} : {}'.format(name, mentions))

        print('All hubs mentions: {}'.format(hubs_count))


def extract_hubs():
    hubs_all = defaultdict(int)
    max_hubs_per_post = -sys.maxsize
    iteration = 0
    with open(POSTS_FILE_PATH, 'r') as f:
        for post in ijson.items(f, 'item'):
            post_hubs = 0
            hubs = post['hubs']
            for hub in hubs:
                hubs_all[hub] += 1
                post_hubs += 1

            if post_hubs > max_hubs_per_post:
                max_hubs_per_post = len(hubs)

            iteration += 1
            if not iteration % SHOW_PROGRESS_EVERY:
                format_str = 'Hubs parsing {} iterations passed'
                print(format_str.format(iteration))

    print('hubs max len : {}'.format(max_hubs_per_post))

    return hubs_all


def main():
    if os.path.exists(HUBS_FILE_PATH):
        print('Hubs data already exist')
        return

    print('Hubs parsing started')
    start_time = time.time()
    hubs = extract_hubs()
    save_hubs(hubs)
    format_str = 'Hubs parsing finished in {} seconds'
    print(format_str.format(time.time() - start_time))


if __name__ == '__main__':
    main()
    show_hubs_info()
