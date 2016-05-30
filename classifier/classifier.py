import pickle
from collections import Counter

from bs4 import BeautifulSoup

from classifier.constants import XGBOOST_PICKLE_CLF, XGBOOST_PICKLE_VECTORIZER
from classifier.dictionary.posts import PostJSONWriter


class Classifier:
    # classifier for external usage
    def __init__(self):
        self.text_parser = PostJSONWriter(write_file=False)
        with open(XGBOOST_PICKLE_CLF, 'rb') as handle:
            self.clf = pickle.load(handle)

        with open(XGBOOST_PICKLE_VECTORIZER, 'rb') as handle:
            self.vectorizer = pickle.load(handle)

    def predict(self, raw_news):
        features = self._transform(raw_news)
        categories = self.clf.predict(features)
        return [int(x) for x in categories]

    def _transform(self, raw_news):
        features = []
        for raw_page in raw_news:
            raw_summary = raw_page['title'] + ' ' + raw_page['summary']
            soup = BeautifulSoup(raw_summary)
            text = soup.get_text()
            feature = self.text_parser.divide_text(text)
            counter = Counter(feature)
            features.append(counter)

        return self.vectorizer.transform(features)
