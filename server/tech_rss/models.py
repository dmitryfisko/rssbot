import feedparser
from django.contrib.postgres import fields
from django.db import models

from rss_parser.utils import get_rss_feeds_from_url


class User(models.Model):
    id = models.PositiveIntegerField(primary_key=True)
    username = models.CharField(max_length=30)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    categories = fields.ArrayField(models.IntegerField())


class Site(models.Model):
    domain = models.CharField(max_length=40, primary_key=True)
    users = models.ManyToManyField(User)
    posts = models.ManyToManyField(User)

    def get_new_news(self):
        feeds = get_rss_feeds_from_url(self.domain)
        if feeds:
            feed = min(feeds, key=len)
            news = feedparser.parse(feed)
            if news:
                return self._filter_old_news(news['entries'])
            else:
                return None
        else:
            return None

    def _filter_old_news(self, all_news):
        news = []
        for page in all_news:
            page_id = self.domain + '_' + page['id']
            was_post = self.post_set.filter(pk=page_id).count()
            if was_post:
                continue

            item = dict()
            item['title'] = page_id['title_detail']['value']
            item['summary'] = page['summary_detail']['value']
            item['url'] = page['link']

            news.append(item)

        return news


class Post(models.Model):
    id = models.CharField(max_length=100)
    category = models.IntegerField()
    url = models.CharField(max_length=100, primary_key=True)
    site = models.ForeignKey(Site, on_delete=models.CASCADE)
