from __future__ import print_function

import json
import os

from constants import RAW_POSTS_FILE_PATH


class PostJSONWriterRAW(object):
    def __init__(self):
        self._file = open(RAW_POSTS_FILE_PATH, 'w')
        self._file.write('[')
        self._post_count = 0

    def process_item(self, post, spider):
        post_str = json.dumps(post) + '\n\n\n,'
        self._file.write(post_str)

        if not post['hubs'] or not post['tags']:
            print(post['url'])
        self._post_count += 1

        return post

    def __del__(self):
        self._file.seek(-1, os.SEEK_END)
        self._file.write(']')
        self._file.close()

        print(self._post_count)
