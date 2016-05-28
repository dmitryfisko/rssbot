from django.apps import AppConfig

from rss_parser.tasks import load_sites_feeds


class RSSParserConfig(AppConfig):
    name = 'rss_parser'

    def ready(self):
        pass
