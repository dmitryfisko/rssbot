# -*- coding: utf8 -*-

import json
import logging

import telepot
from django.template.loader import render_to_string
from django.http import HttpResponseForbidden, HttpResponseBadRequest, JsonResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

from .utils import parse_planetpy_rss, create_user, send_inline_add_filters, send_inline_remove_filters, \
    send_notification_added_filter, send_notification_removed_filter, send_site_reading_started, \
    send_site_reading_stopped_inline, display_help, send_notification_reading_stopped

TelegramBot = telepot.Bot(settings.TELEGRAM_BOT_TOKEN)

logger = logging.getLogger('telegram.bot')


def _display_planetpy_feed():
    return render_to_string('feed.md', {'items': parse_planetpy_rss()})


def _add_site():
    return render_to_string('feed.md', {'items': parse_planetpy_rss()})


def _remove_site():
    return render_to_string('feed.md', {'items': parse_planetpy_rss()})


def _add_category():
    return render_to_string('feed.md', {'items': parse_planetpy_rss()})


def _remove_category():
    return render_to_string('feed.md', {'items': parse_planetpy_rss()})


class CommandReceiveView(View):
    @staticmethod
    def post(request, bot_token):
        if bot_token != settings.TELEGRAM_BOT_TOKEN:
            return HttpResponseForbidden('Invalid token')

        raw = request.body.decode('utf-8')
        logger.info(raw)

        try:
            payload = json.loads(raw)
        except ValueError:
            return HttpResponseBadRequest('Invalid request body')
        else:
            msg = payload.get('message')
            user_id = cmd = query_id = callback_data = None
            if msg:
                user_id = msg['from']['id']
                cmd = payload.get('text')  # command
            else:
                callback_data = msg.get('data')
                query_id = payload.get('id')

            if isinstance(cmd, str):
                command = cmd.split()[0].lower()
            else:
                command = None

            if command == '/start':
                create_user(user_id, payload)
                display_help(TelegramBot, user_id)
            elif command == '/help':
                display_help(TelegramBot, user_id)
            elif command == '/read':
                send_site_reading_started(TelegramBot, cmd, user_id)
            elif command == '/stop':
                send_site_reading_stopped_inline(TelegramBot, user_id)
            elif command == '/addfilter':
                send_inline_add_filters(TelegramBot, user_id)
            elif command == '/remfilter':
                send_inline_remove_filters(TelegramBot, user_id)
            elif callback_data:
                data_command, second_param = callback_data.split('_')
                if data_command == 'addfilter':
                    filter_num = int(second_param)
                    send_notification_added_filter(TelegramBot, user_id, query_id, filter_num)
                elif data_command == 'remfilter':
                    filter_num = int(second_param)
                    send_notification_removed_filter(TelegramBot, user_id, query_id, filter_num)
                elif data_command == 'stop':
                    domain = second_param
                    send_notification_reading_stopped(TelegramBot, user_id, query_id, domain)
            else:
                TelegramBot.sendMessage(user_id, 'I do not understand you, Sir!')

        return JsonResponse({}, status=200)

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super(CommandReceiveView, self).dispatch(request, *args, **kwargs)
