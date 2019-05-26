import chess.pgn
import chess.engine as engine
import chess
import pandas as pd
import numpy as np
from math import ceil


def get_games(filename):
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)
        yield game
        '''
        cnt =1
        while game:
            cnt +=1
            if cnt==3:
                 yield game
            game = chess.pgn.read_game(pgn)
        '''

def main():
# Создать pgn самостоятельно!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    engine = chess.engine.SimpleEngine.popen_uci(r'C:\Users\Asus\PredictingRatings\stockfish-10-win\Windows\stockfish_10_x64.exe')

    # Из extracting.py
    games = get_games(r'C:\Users\Asus\PredictingRatings\data\test1.pgn')

    white_elos = []
    black_elos = []
    results = []
    moves = []
    uci_moves = []
    counts = []
    scores=[]
    move_scores=[]

    for game in games:
        print(game)
        if 'WhiteElo' in game.headers:
            white_elos.append(game.headers['WhiteElo'])
        if 'BlackElo' in game.headers:
            black_elos.append(game.headers['BlackElo'])
        results.append(game.headers['Result'])

        node = game.variation(0)
        sans = []
        uci = []
        count = 0

        while node.variations:
            board = node.board()
            count += 1
            sans.append(node.san())
            uci.append(node.uci())
            if count >0:
                an_scores = engine.analyse(board, chess.engine.Limit(time=100,nodes=1,depth=15))
                score = int(str(an_scores['score'].pov(color=True)).strip('#'))
                scores.append(score)

            node = node.variations[0]

        board = node.board()
        count += 1
        sans.append(node.san())
        uci.append(node.uci())
        count = ceil(count / 2)
        an_scores = engine.analyse(board, chess.engine.Limit(time=100,nodes=1,depth=15))

        score = int(str(an_scores['score'].pov(color=True)).strip('#'))
        scores.append(score)
        scores.append(score)
        move_scores.append(scores)

        engine.quit()

        counts.append(count)
        moves.append(sans)
        uci_moves.append(uci)

        print(scores)
        print('Results',len(results))
        print('Moves',len(moves))
        print('Uci',len( uci_moves))
        print('Scores',len(move_scores))
        print('Counts',len(counts))
        print(counts)

        data=pd.DataFrame(np.column_stack(results),columns=['Result'])
        data['Moves']=moves
        data['UCI']=uci_moves
        data['Scores']=move_scores
        data['NumMoves']=counts

        elos = pd.DataFrame(np.column_stack([white_elos, black_elos]),
                            columns=['WhiteElo', 'BlackElo'])

        # Из exploring.ipynb

if __name__ == "__main__":
    main()