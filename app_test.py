import chess.pgn

def get_rating(filename):
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)

    result='Ваш рейтинг...ОНО РАБОТАЕТ!!!!'
    return result