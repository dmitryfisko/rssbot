from __future__ import absolute_import

from datetime import timedelta

import telepot
from celery.task import periodic_task
from django.conf import settings
from multiprocessing import Lock

from classifier.classifier import Classifier
from classifier.constants import CATEGORIES_SHORT
from rss_parser.utils import single_instance_task, send_post_to_subscribers, fix_multiprocessing

TelegramBot = telepot.Bot(settings.TELEGRAM_BOT_TOKEN)


@periodic_task(run_every=timedelta(seconds=5), ignore_result=True)
def keep_alive_connection():
    pass


def save_post(category, page, site):
    from tech_rss.models import Post

    post_id = page['id']
    title = page['title']
    url = page['url']
    post = Post(id=post_id, category=category,
                title=title, url=url)
    post.save()
    site.posts.add(post)
    return url, title


@periodic_task(run_every=timedelta(minutes=30), ignore_result=True)
def load_sites_feeds():
    from tech_rss.models import Site
    fix_multiprocessing()

    clf = Classifier()
    for site in Site.objects.all():
        print('Starting {}'.format(site.domain))
        news = site.get_new_news()

        if not news:
            continue

        categories = clf.predict(news)
        for category, page in zip(categories, news):
            print(CATEGORIES_SHORT[category])
            print(page['title'], '\n')

            url, title = save_post(category, page, site)

            users = site.users.filter(categories__contains=[category])
            users_id = [getattr(user, 'id') for user in users]

            send_post_to_subscribers(TelegramBot, users_id, url, title)
