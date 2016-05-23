import json

import telepot

token = '188482013:AAHXx0jVJnKveG-bmLvuZD9ihk8oes5UMHs'
TelegramBot = telepot.Bot(token)
print(json.dumps(TelegramBot.getUpdates()))
print(TelegramBot.getMe())
