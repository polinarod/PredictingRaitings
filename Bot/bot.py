import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
#import telegram.files as tfile
#import app_test

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Обработка команд
def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Отправьте мне Ваши партии в формате pgn')

def textMessage(bot, update):
    response = 'Пожалуйста, отправьте партии'
    bot.send_message(chat_id=update.message.chat_id, text=response)

def fileMessage(bot, update):
    response = 'Обрабатываю Ваши партии...'
    bot.send_message(chat_id=update.message.chat_id, text=response)
    test_response='Просто посмотреть, что присылает на отправку файла' + update.message.text
    bot.send_message(chat_id=update.message.chat_id, text=test_response)
    #app.get_rating(update.message.docu)

def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    updater = Updater(token='846868641:AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw') # Токен API к Telegram
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