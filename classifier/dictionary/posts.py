# coding=utf-8
import json
import os
import re

import ijson
import pymorphy2

from constants import POSTS_FILE_PATH, RAW_POSTS_FILE_PATH


class PostJSONWriter(object):
    def __init__(self):
        self._file = open(POSTS_FILE_PATH, 'w')
        self._file.write('[')
        self._regex_broad = re.compile(ur'[а-яёА-ЯЁa-zA-Z0-9_\-\+]+', re.U)
        self._regex_sym = re.compile(ur'^[а-яёА-ЯЁa-zA-Z]+.*$', re.U)
        self._morph = pymorphy2.MorphAnalyzer()

    def divide_text(self, text):
        norm_tokens = []
        tokens = self._regex_broad.findall(text.lower())
        for token in tokens:
            if not re.match(self._regex_sym, token):
                continue
            token = self._morph.parse(token)[0].normal_form
            # token = en.lemma(token)
            norm_tokens.append(token)

        return norm_tokens

    def write_item(self, post):
        post['title'] = self.divide_text(post['title'])
        post['content'] = self.divide_text(post['content'])
        post['tags'] = self.divide_text(' '.join(post['tags']))
        post['hubs'] = [hub.lower() for hub in post['hubs']]

        post_json = json.dumps(post) + '\n\n\n,'
        self._file.write(post_json)

    def __del__(self):
        self._file.seek(-1, os.SEEK_END)
        self._file.write(']')
        self._file.close()


def main():
    if os.path.exists(POSTS_FILE_PATH):
        print('Processed posts already exists')
        return

    print('Posts processing started')
    writer = PostJSONWriter()
    with open(RAW_POSTS_FILE_PATH, 'r') as f:
        for post in ijson.items(f, 'item'):
            writer.write_item(post)

    print('Posts processing finished')


if __name__ == '__main__':
    main()
