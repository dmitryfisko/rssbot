from time import sleep

import environ
import telepot

env = environ.Env(DEBUG=(bool, False))
environ.Env.read_env()

bot_token = 'BOT_TOKEN'
bot = telepot.Bot(bot_token)
print(bot.getMe())

print(bot.setWebhook(url='https://rssbot.ml/tech_rss/bot/{bot_token}/'.format(bot_token=bot_token)))
