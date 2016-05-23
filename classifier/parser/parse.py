import logging
import os
import time
from scrapy.crawler import CrawlerProcess

from constants import CRAWLER_LOG_PATH, RAW_POSTS_FILE_PATH
from spiders.tmspider import TMSpider


def get_settings():
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'ITEM_PIPELINES': {
            'parser.pipelines.PostJSONWriterRAW': 1
        },
        'RETRY_TIMES': 5,
        'LOG_FILE': CRAWLER_LOG_PATH,
        'LOG_LEVEL': logging.INFO
    })

    return process


def main():
    if os.path.exists(RAW_POSTS_FILE_PATH):
        print('Crawler data already exist')
        return

    print('Crawler started his work')
    start_time = time.time()
    process = get_settings()
    process.crawl(TMSpider)
    process.start()
    process.join()
    format_str = 'Crawling finished in {} seconds'
    print(format_str.format(time.time() - start_time))


if __name__ == "__main__":
    main()
