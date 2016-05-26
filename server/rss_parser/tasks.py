from __future__ import absolute_import
from datetime import timedelta

import telepot
from celery.task import periodic_task
from django.conf import settings

TelegramBot = telepot.Bot(settings.TELEGRAM_BOT_TOKEN)


@periodic_task(run_every=timedelta(minutes=5))
def keep_alive_connection():
    TelegramBot.getMe()
