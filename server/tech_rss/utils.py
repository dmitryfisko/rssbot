# -*- coding: utf8 -*-
import urllib
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from xml.etree import cElementTree

import requests
from bs4 import BeautifulSoup
from django.template.loader import render_to_string
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton

from tech_rss.models import User, Site

CATEGORIES_SHORT = [
    'Веб разработка',
    'Безопасность',
    'Программирование',
    'Алгоритмы',
    'Системное программирование',
    'Данные',
    'Железо',
    'Сеть',
    'Администрирование',
    'Мобильные приложения',
    'Математика',
    'Тестирование',
    'Машинное обучение',
    'Бизнес',
    'Законодательство',
    'Научно-популярное',
    'Дизайн'
]


def display_help(bot, user_id):
    text = render_to_string('help.md')
    bot.sendMessage(user_id, text=text, parse_mode='Markdown')


def create_user(user_id, msg):
    username = msg['from']['username']
    first_name = msg['from']['first_name']
    last_name = msg['from']['last_name']
    categories = list(range(len(CATEGORIES_SHORT)))

    user = User(id=user_id, username=username,
                first_name=first_name, last_name=last_name,
                categories=categories)
    user.save()
    return user


def create_site(domain):
    site = Site(domain=domain)
    site.save()
    return site


def _split_buttons_on_lines(buttons, line_len=2):
    lines_num = (len(buttons) + line_len - 1) // line_len
    buttons = [buttons[i * line_len:(i + 1) * line_len]
               for i in range(lines_num)]
    return buttons


def _send_inline_filters(is_add, bot, user_id):
    user = User.objects.get(pk=user_id)
    buttons = []
    for ind, category in enumerate(CATEGORIES_SHORT):
        callback_data = None
        if is_add and ind not in user.categories:
            callback_data = 'addfilter_' + str(ind)
        if not is_add and ind in user.categories:
            callback_data = 'remfilter_' + str(ind)

        if callback_data:
            button = InlineKeyboardButton(text=category, callback_data=callback_data)
            buttons.append(button)

    if buttons:
        buttons = _split_buttons_on_lines(buttons)
        markup = InlineKeyboardMarkup(inline_keyboard=buttons)

        if is_add:
            bot.sendMessage(user_id, 'Добавить фильтры:', reply_markup=markup)
        else:
            bot.sendMessage(user_id, 'Удалить фильтры:', reply_markup=markup)
    else:
        if is_add:
            bot.sendMessage(user_id, 'Вы уже добавили все возможные фильтры.')
        else:
            bot.sendMessage(user_id, 'Список активных фильтров пуст.')


def send_inline_add_filters(bot, user_id):
    is_add = True
    _send_inline_filters(is_add, bot, user_id)


def send_inline_remove_filters(bot, user_id):
    is_add = False
    _send_inline_filters(is_add, bot, user_id)


def send_notification_added_filter(bot, user_id, query_id, filter_num):
    assert 0 <= filter_num < len(CATEGORIES_SHORT)

    user = User.objects.get(pk=user_id)
    if filter_num in user.categories:
        bot.answerCallbackQuery(query_id, text='Фильтр уже был добавлен.')
    else:
        user.categories.append(filter_num)
        user.save()
        bot.answerCallbackQuery(query_id, text='Фильтр успешно добавлен.')


def send_notification_removed_filter(bot, user_id, query_id, filter_num):
    assert 0 <= filter_num < len(CATEGORIES_SHORT)

    user = User.objects.get(pk=user_id)
    if filter_num not in user.categories:
        bot.answerCallbackQuery(query_id, text='Фильтр уже был удалён.')
    else:
        user.categories.remove(filter_num)
        user.save()
        bot.answerCallbackQuery(query_id, text='Фильтр успешно удалён.')


def get_rss_feeds_from_url(url):
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


def send_site_reading_started(bot, command, user_id):
    command_parts = len(command.split())
    if command_parts != 2:
        if command_parts < 2:
            text = 'Не указан адресс сайта.'
        else:
            text = 'Указано несколько сайтов.'

        text += '\nПример: */add habr.ru*'
        bot.sendMessage(user_id, text=text, parse_mode='Markdown')
        return

    url = command.split()[1]
    if 'http' not in url:
        url = 'http://' + url
    domain = urlparse(url).netloc
    if not domain:
        text = 'Указан неправильный адресс сайта.'
        text += '\nПример: */add habr.ru*'
        bot.sendMessage(user_id, text=text, parse_mode='Markdown')
        return

    # does check url for network status?

    is_site_added = Site.objects.filter(pk=domain).count() != 0
    if not is_site_added:
        if not get_rss_feeds_from_url(url):
            text = 'Указанный сайт не поддерживает RSS.'
            bot.sendMessage(user_id, text=text, parse_mode='Markdown')
            return

    domain = urlparse(url).netloc
    user = User.objects.get(pk=user_id)
    if user.site_set.filter(domain=domain).count():
        text = 'Вы уже добавляли этот сайт.'
        bot.sendMessage(user_id, text=text)
        # bot.answerCallbackQuery(query_id, text=text)
    else:
        if is_site_added:
            site = Site.objects.get(pk=domain)
        else:
            site = create_site(domain)
        site.users.add(user)

        text = 'Сайт успешно добавлен.'
        bot.sendMessage(user_id, text=text)
        # bot.answerCallbackQuery(query_id, text=text)


def send_site_reading_stopped_inline(bot, user_id):
    user = User.objects.get(pk=user_id)
    domains = [getattr(site, 'domain') for site in user.site_set.all()]
    buttons = []
    for ind, domain in enumerate(domains):
        callback_data = 'stop_' + domain
        button = InlineKeyboardButton(text=domain, callback_data=callback_data)
        buttons.append(button)

    if buttons:
        buttons = _split_buttons_on_lines(buttons, line_len=2)
        markup = InlineKeyboardMarkup(inline_keyboard=buttons)

        bot.sendMessage(user_id, 'Перестать читать:', reply_markup=markup)
    else:
        bot.sendMessage(user_id, 'В данный момент у вас нет читаемых сайтов.')


def send_notification_reading_stopped(bot, user_id, query_id, domain):
    user = User.objects.get(pk=user_id)
    is_read_domain = user.site_set.filter(domain=domain).count()
    if is_read_domain:
        site = Site.objects.get(pk=domain)
        site.users.remove(user)
        site.save()
        bot.answerCallbackQuery(query_id, text='Сайт успешно удалён из читаемых.')
    else:
        bot.answerCallbackQuery(query_id, text='Сайт уже был удалён из читаемых.')


def parse_planetpy_rss():
    """Parses first 10 items from http://planetpython.org/rss20.xml
    """
    response = requests.get('http://planetpython.org/rss20.xml')
    parsed_xml = cElementTree.fromstring(response.content)
    items = []

    for node in parsed_xml.iter():
        if node.tag == 'item':
            item = {}
            for item_node in list(node):
                if item_node.tag == 'title':
                    item['title'] = item_node.text
                if item_node.tag == 'link':
                    item['link'] = item_node.text

            items.append(item)

    return items[:10]
