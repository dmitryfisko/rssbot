import os

from dictionary import words
from dictionary import hubs
from dictionary import posts
from model.news import news_input_gen
from parser import parse


def main():
    os.chdir('parser')
    parse.main()

    os.chdir('../dictionary')
    posts.main()
    words.main()
    hubs.main()

    os.chdir('../')
    news_input_gen.main()


if __name__ == '__main__':
    main()
