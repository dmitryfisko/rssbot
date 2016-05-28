from __future__ import absolute_import

from datetime import timedelta

import telepot
from celery.task import periodic_task
from django.conf import settings

from rss_parser.utils import single_instance_task

TelegramBot = telepot.Bot(settings.TELEGRAM_BOT_TOKEN)


@periodic_task(run_every=timedelta(seconds=5), ignore_result=True)
def keep_alive_connection():
    pass


@periodic_task(run_every=timedelta(seconds=60), ignore_result=True)
@single_instance_task(60 * 60 * 2)
def load_sites_feeds():
    from tech_rss.models import Site

    clf = Classifier()
    for site in Site.objects.all():
        news = site.get_new_news()

        if not news:
            continue

        for page in news:
            category
