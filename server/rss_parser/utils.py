import functools
import re
import urllib
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from celery.signals import worker_process_init
from django.core.cache import cache

from bs4 import BeautifulSoup


def single_instance_task(timeout):
    def task_exc(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lock_id = "celery-single-instance-" + func.__name__
            acquire_lock = lambda: cache.add(lock_id, "true", timeout)
            release_lock = lambda: cache.delete(lock_id)
            if acquire_lock():
                try:
                    func(*args, **kwargs)
                finally:
                    release_lock()

        return wrapper

    return task_exc


def send_post_to_subscribers(bot, users_id, url, title):
    text = title + '\n' + url
    for user_id in users_id:
        bot.sendMessage(user_id, text=text)


@worker_process_init.connect
def fix_multiprocessing(**kwargs):
    from multiprocessing import current_process
    try:
        current_process()._config
    except AttributeError:
        current_process()._config = {'semprefix': '/mp'}


def get_rss_feeds_from_url(url):
    schema = '^https?://'
    if not re.match(schema, url):
        url = 'http://' + url
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/35.0.1916.47 Safari/537.36'
            }
        )
        page = urlopen(req).read()
        soup = BeautifulSoup(page, "lxml")

        rss_links = soup.findAll('link', type='application/rss+xml')
        rss_feeds = [link['href'] for link in rss_links]
        if len(rss_feeds) > 0:
            return rss_feeds
        else:
            return None
    except HTTPError as _:
        return None
    except URLError as _:
        return None
