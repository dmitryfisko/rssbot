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


class Post(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    category = models.IntegerField()
    title = models.CharField(max_length=250)
    url = models.CharField(max_length=100)


class Site(models.Model):
    domain = models.CharField(max_length=40, primary_key=True)
    users = models.ManyToManyField(User)
    posts = models.ManyToManyField(Post)

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
            was_post = Post.objects.filter(pk=page_id).count()
            if was_post:
                continue

            item = dict()
            item['id'] = page_id
            item['title'] = page['title_detail']['value']
            item['summary'] = page['summary_detail']['value']
            item['url'] = page['link']

            news.append(item)

        return news
