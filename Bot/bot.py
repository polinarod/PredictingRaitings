import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import requests
from app import get_rating
from app_test import test_rating

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

token='846868641:AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw'

# Обработка команд
def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Отправьте мне Ваши партии в формате pgn')

def textMessage(bot, update):
    response = 'Пожалуйста, отправьте партии'
    response=update.message.text
    bot.send_message(chat_id=update.message.chat_id, text=response)


def fileMessage(bot, update):
    response = 'Обрабатываю Ваши партии...'
    bot.send_message(chat_id=update.message.chat_id, text=response)
    pgn_id=update.message.document.file_id

    file = bot.get_file(pgn_id)
    file.download('game.pgn')

    get_rating('game.pgn')

   # test_response=requests.get('https://api.telegram.org/file/bot'+token+'/getFile?file_id='+pgn_id)
    #https: // api.telegram.org / bot846868641: AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw / documents / file_0.pgn
   # bot.send_message(chat_id=update.message.chat_id, text=str(test_response))


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    updater = Updater(token=token) # Токен API к Telegram
    dispatcher = updater.dispatcher
    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, textMessage)
    file_message_handler=MessageHandler(Filters.document,fileMessage)


    # Добавляем хендлеры в диспетчер
    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)
    dispatcher.add_handler(file_message_handler)

    # Начинаем поиск обновлений
    updater.start_polling(clean=True)
    print('Im working!')

    # Останавливаем бота, если были нажаты Ctrl + C
    updater.idle()
    print('I was stopped')


if __name__ == '__main__':
    main()