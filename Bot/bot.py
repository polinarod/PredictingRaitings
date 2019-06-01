import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Обработка команд
def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Привет, давай пообщаемся?')

def textMessage(bot, update):
    response = 'Получил Ваше сообщение: ' + update.message.text
    bot.send_message(chat_id=update.message.chat_id, text=response)

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    REQUEST_KWARGS = {
        'proxy_url': 'socks5://159.203.118.239:8080'
         #Optional, if you need authentication:
         #,'username': 'mochajs',
        #'password': 'topoliniypuh',
    }

    updater = Updater(token='846868641:AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw',request_kwargs=REQUEST_KWARGS) # Токен API к Telegram
    #updater = Updater(token='846868641:AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw')
    dispatcher = updater.dispatcher
    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, textMessage)

    # Добавляем хендлеры в диспетчер
    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)

    # Начинаем поиск обновлений
    updater.start_polling(clean=True)
    print('Im working!')

    # Останавливаем бота, если были нажаты Ctrl + C
    updater.idle()
    print('I was stopped')


if __name__ == '__main__':
    main()