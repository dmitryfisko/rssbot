# coding=utf-8
from os.path import expanduser

USER_HOME = expanduser("~")
PROJECT_DATA_FOLDER = USER_HOME + '/univ/techbot/classifier/data/'
MODELS_DATA_FOLDER = USER_HOME + '/univ/techbot/classifier/models/'
POSTS_FILE_PATH = PROJECT_DATA_FOLDER + 'posts.json'
RAW_POSTS_FILE_PATH = PROJECT_DATA_FOLDER + 'raw_posts.json'
CRAWLER_LOG_PATH = PROJECT_DATA_FOLDER + 'crawler.log'
WORDS_FILE_PATH = PROJECT_DATA_FOLDER + 'words.json'
HUBS_FILE_PATH = PROJECT_DATA_FOLDER + 'hubs.json'

TF_ONE_SHOT_TRAIN_FILE_PATH = PROJECT_DATA_FOLDER + 'data_bagofwords_train.tfrecords'
TF_ONE_SHOT_EVAL_FILE_PATH = PROJECT_DATA_FOLDER + 'data_bagofwords_eval.tfrecords'

TF_TFIDF_TRAIN_FILE_PATH = PROJECT_DATA_FOLDER + 'data_tfidf7_train.tfrecords'
TF_TFIDF_EVAL_FILE_PATH = PROJECT_DATA_FOLDER + 'data_tfidf7_eval.tfrecords'
TF_TFIDF_VECTORIZER_FILE_PATH = PROJECT_DATA_FOLDER + 'data_tfidf7_vectorizer.pickle'

XGBOOST_PICKLE_CLF = PROJECT_DATA_FOLDER + 'xgboost_classifier_mid.pickle'
XGBOOST_PICKLE_VECTORIZER = PROJECT_DATA_FOLDER + 'xgboost_vectorizer_mid.pickle'

CATEGORIES = [
    [u'разработка веб-сайтов', u'браузеры', u'jquery', u'html', u'javascript', u'php', u'монетизация веб-сервисов',
     u'веб-аналитика'],  # Веб разработка
    [u'информационная безопасность', u'криптография'],  # Безопасность
    [u'программирование', u'параллельное программирование', u'промышленное программирование'],  # Программирование
    [u'алгоритмы', u'спортивное программирование', u'серверная оптимизация', u'клиентская оптимизация'],  # Алгоритмы
    [u'системное программирование', u'операционные системы'],  # Системное программирование
    [u'суперкомпьютеры', u'облачные вычисления', u'data mining', u'виртуализация', u'поисковые технологии',
     u'высокая производительность'],  # Данные
    [u'гаджеты', u'железо', u'робототехника', u'смартфоны', u'старое железо', u'электроника для начинающих',
     u'видеокарты'],  # Железо
    [u'беспроводные технологии', u'сетевые технологии', u'сетевое оборудование', u'децентрализованные сети'],  # Сеть
    [u'системное администрирование', u'администрирование баз данных', u'linux', u'настройка linux', u'хостинг',
     u'серверное администрирование', u'ит-инфраструктура'],  # Администрирование
    [u'разработка мобильных приложений', u'аналитика мобильных приложений', u'разработка под windows phone',
     u'android', u'ios', u'разработка под android', u'разработка под ios'],  # Мобильные приложения
    [u'математика'],  # Математика
    [u'тестирование it-систем', u'тестирование веб-сервисов', u'отладка'],  # Тестирование
    [u'искусственный интеллект', u'визуализация данных', u'машинное обучение'],  # Машинное обучение
    [u'growth hacking', u'бизнес-модели', u'монетизация it-систем', u'финансы в it', u'управление продажами',
     u'управление проектами', u'финансы в it-индустрии', u'управление продуктом', u'развитие стартапа',
     u'платежные системы', u'венчурные инвестиции'],  # Бизнес
    [u'регулирование интернета', u'патентование', u'законодательство и it-бизнес',
     u'it-стандарты'],  # Законодательство
    [u'научно-популярное'],  # Научно-популярное
    [u'дизайн в it', u'интерфейсы', u'веб-дизайн']  # Дизайн
]

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
