import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from app import get_rating

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

token='846868641:AAFtwHuP6yme_nRtAfgIIPilVNyQcEXaTgw'
file=None

def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Отправьте мне Ваши партии в формате pgn')

def helpCommand(bot,update):
    response='/get_names - получить имена игроков (только после отправки файла)\n/get_current - получить рейтинги игроков на момент партии (только после отправки файла)'
    bot.send_message(chat_id=update.message.chat_id, text=response)

def textMessage(bot, update):
    response = 'Пожалуйста, отправьте партии'
    bot.send_message(chat_id=update.message.chat_id, text=response)

def getNamesCommand(bot,update):
    result=get_rating('game.pgn','names')
    bot.send_message(chat_id=update.message.chat_id, text=result)

def getCurrentCommand(bot,update):
    result=get_rating('game.pgn','current')
    bot.send_message(chat_id=update.message.chat_id, text=result)

def fileMessage(bot, update):
    response = 'Обрабатываю Ваши партии...'
    bot.send_message(chat_id=update.message.chat_id, text=response)
    pgn_id=update.message.document.file_id
    file = bot.get_file(pgn_id)
    file.download('game.pgn')

    result=get_rating('game.pgn','ratings')
    bot.send_message(chat_id=update.message.chat_id, text=result)


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    updater = Updater(token=token)
    dispatcher = updater.dispatcher
    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, textMessage)
    file_message_handler=MessageHandler(Filters.document,fileMessage)
    get_names_command_handler=CommandHandler('get_names',getNamesCommand)
    get_rating_command_handler=CommandHandler('get_current',getCurrentCommand)
    help_command_handler=CommandHandler('help',helpCommand)

    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)
    dispatcher.add_handler(file_message_handler)
    dispatcher.add_handler(get_names_command_handler)
    dispatcher.add_handler(get_rating_command_handler)
    dispatcher.add_handler(help_command_handler)

    updater.start_polling(clean=True)
    print('Im working!')

    updater.idle()

if __name__ == '__main__':
    main()