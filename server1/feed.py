from urllib.request import urlopen

from bs4 import BeautifulSoup

import urllib

req = urllib.request.Request(
    'http://sdfsdsfsdfsdgfgdf.com/',
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
print(rss_feeds)
