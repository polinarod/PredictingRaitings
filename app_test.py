import chess.pgn

def get_rating(filename):
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)

    if 'WhiteElo' in game.headers:
        white_elo= game.headers['WhiteElo']
    if 'BlackElo' in game.headers:
        black_elo=game.headers['BlackElo']

    result='Рейтинг белых: '+white_elo+'\nРейтинг черных: '+black_elo
    return result