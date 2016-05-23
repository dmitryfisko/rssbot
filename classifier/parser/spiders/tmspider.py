# coding=utf-8
from bs4 import BeautifulSoup
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from scrapy.spiders import CrawlSpider
from scrapy.spiders import Rule


class TMSpider(CrawlSpider):
    name = "TM company sites parser"
    allowed_domains = ["habrahabr.ru", "megamozg.ru", "geektimes.ru"]
    start_urls = ['https://habrahabr.ru/hubs/',
                  'https://megamozg.ru/hubs/',
                  'https://geektimes.ru/hubs/']
    rules = (
        Rule(LinkExtractor(allow=('/hubs/$', '/hubs/page[0-9]*/$')), follow=True),
        Rule(LinkExtractor(allow=('/hub/[a-z0-9_-]*/$',
                                  '/hub/[a-z0-9_-]*/page[0-9]*/$')), follow=True),
        Rule(LinkExtractor(allow='/post/[0-9]*/',
                           deny='//div[not(@class="post_title")]'), callback='parse_post')
    )

    @staticmethod
    def parse_post(response):
        post = dict()
        sel = Selector(response)

        title = sel.xpath('//h1[@class="title"]/'
                          'span[@class="post_title"]/text()').extract_first()
        post["title"] = title

        # hubs = sel.xpath('//div[@class="hubs"]/span[@class="profiled_hub"]/'
        #                  'preceding-sibling::a/text()').extract()
        hubs = sel.xpath('//div[@class="hubs"]/a/text()').extract()
        post["hubs"] = hubs

        tags = sel.xpath('//ul[@class="tags icon_tag"]/li/a/text()').extract()
        post['tags'] = tags

        content_html = sel.select('//div[@class="content html_format"]').extract_first()
        soup = BeautifulSoup(content_html)
        for match in soup.findAll('code'):
            match.replaceWith('')
        post['content'] = soup.get_text()

        post['url'] = response.url

        return post
