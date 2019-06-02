import chess.pgn
import chess.engine as engine
import chess
import pandas as pd
import numpy as np
from math import ceil
from collections import defaultdict, Counter
import pickle
import scipy.stats

def get_rating(filename,mode='ratings'):
    engine = chess.engine.SimpleEngine.popen_uci(r'C:\Users\Asus\PredictingRatings\stockfish-10-win\Windows\stockfish_10_x64.exe')

    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)

    moves = []
    uci_moves = []
    scores=[]
    move_scores=[]

    if mode=='names':
        white=game.headers['White']
        black = game.headers['Black']
        names = 'Белые: ' + str(white) + '\nЧерные: ' + str(black)
        return names


    if 'WhiteElo' in game.headers:
        white_elo=int(int(game.headers['WhiteElo']))
    if 'BlackElo' in game.headers:
        black_elo=int(game.headers['BlackElo'])
    result=game.headers['Result']

    if mode=='current':
        ratings = 'Текущий рейтинг белых: ' + str(white_elo) + '\nТекущий рейтинг черных: ' + str(black_elo)
        return ratings

    node = game.variation(0)
    sans = []
    uci = []
    count = 0

    while node.variations:
        board = node.board()
        count += 1
        sans.append(node.san())
        uci.append(node.uci())
        if count > 0:
            an_scores = engine.analyse(board, chess.engine.Limit(time=100, nodes=1, depth=15))
            score = int(str(an_scores['score'].pov(color=True)).strip('#'))
            scores.append(score)

        node = node.variations[0]

    board = node.board()
    count += 1
    sans.append(node.san())
    uci.append(node.uci())
    count = ceil(count / 2)
    an_scores = engine.analyse(board, chess.engine.Limit(time=100, nodes=1, depth=15))

    score = int(str(an_scores['score'].pov(color=True)).strip('#'))
    scores.append(score)

    engine.quit()
    moves.append(sans)
    uci_moves.append(uci)
    move_scores.append(scores)

    data=pd.DataFrame(columns=['Result'])
    data['Result']=result
    data['Moves'] = moves
    data['UCI'] = uci_moves
    data['Scores'] = move_scores
    data['NumMoves'] = count
    data['WhiteElo'] = white_elo
    data['BlackElo'] = black_elo
    data['MeanScore'] = data.Scores.apply(lambda x: np.mean(x))
    data['ModeScore'] = data.Scores.apply(lambda x: scipy.stats.mode(x))
    data['FinalScore'] = data.Scores.apply(lambda x: x[-1])

    features = data.copy()

    features.Result = features.Result.apply(lambda x: 0.5 if x == '1/2-1/2' else x)
    features.Result = features.Result.apply(lambda x: 1 if x == '1-0' else x)
    features.Result = features.Result.apply(lambda x: 0 if x == '0-1' else x)

    features['FullMoves'] = features.Moves.apply(lambda x: len(x))

    features = features.drop(columns=['Moves', 'Scores', 'UCI', 'WhiteElo', 'BlackElo'])

    data['WhiteMoves'] = data.Moves.apply(lambda x: x[::2])
    data['WhiteMovesUCI'] = data.UCI.apply(lambda x: x[::2])
    data['WhiteScores'] = data.Scores.apply(lambda x: x[::2])
    data['BlackMoves'] = data.Moves.apply(lambda x: x[1::2])
    data['BlackMovesUCI'] = data.UCI.apply(lambda x: x[1::2])
    data['BlackScores'] = data.Scores.apply(lambda x: x[1::2])
    data['DiffScores'] = data.Scores.apply(lambda x: np.diff(x).tolist())
    data['DiffWhiteScores'] = data.DiffScores.apply(lambda x: x[::2])
    data['DiffBlackScores'] = data.DiffScores.apply(lambda x: x[1::2])

    data = data.drop(columns=['NumMoves', 'MeanScore', 'ModeScore', 'FinalScore'])

    data = data.append(pd.DataFrame(columns=["DebScores", "MitScores", "EndScores"]), sort=True)

    data['DebScores'] = data.Scores.apply(lambda x: x[:16])
    data['MitScores'] = data.Scores.apply(lambda x: x[16:36])
    data['EndScores'] = data.Scores.apply(lambda x: x[36:])

    modes = ['DebScores', 'MitScores', 'EndScores']
    var = ['White', 'Black', 'Diff', 'DiffWhite', 'DiffBlack']

    for m in modes:
        data['White' + m] = data[m].apply(lambda x: x[::2])
        data['Black' + m] = data[m].apply(lambda x: x[1::2])
        data['Diff' + m] = data[m].apply(lambda x: np.diff(x).tolist())
        data['DiffWhite' + m] = data['Diff' + m].apply(lambda x: x[::2])
        data['DiffBlack' + m] = data['Diff' + m].apply(lambda x: x[1::2])

    stats = [np.min, np.max, np.mean, lambda x: np.median(np.abs(x)), np.std,
             lambda x: round(np.var(x), 2), lambda x: max(set(x), key=x.count)]
    stat_names = ['Min', 'Max', 'Mean', 'Median', 'Std', 'Variance', 'Mode']
    color = ['White', 'Black']

    for stat, stat_name in zip(stats, stat_names):
        features[stat_name + 'Score'] = data.Scores.apply(lambda x: stat(x))

    for c in color:
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + c + 'Score'] = data[c + 'Scores'].apply(lambda x: stat(x))

        features['Final' + c + 'Score'] = data[c + 'Scores'].apply(lambda x: x[-1])

    for stat, stat_name in zip(stats, stat_names):
        features[stat_name + 'DeltaScore'] = data.DiffScores.apply(lambda x: stat(x))

    for c in color:
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + c + 'DeltaScore'] = data['Diff' + c + 'Scores'].apply(lambda x: stat(x))

    modes = ['Deb', 'Mit', 'End']

    for m in modes:
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + m + 'Score'] = data[m + 'Scores'].apply(lambda x: stat(x) if len(x) != 0 else -1)

    for c in color:
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + c + m + 'Score'] = data[c + m + 'Scores'].apply(
                lambda x: stat(x) if len(x) != 0 else -1)

        features['Final' + c + m + 'Score'] = data[c + m + 'Scores'].apply(lambda x: x[-1] if len(x) != 0 else 0)

    for m in modes:
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + m + 'DeltaScore'] = data['Diff' + m + 'Scores'].apply(
                lambda x: stat(x) if len(x) != 0 else -1)

    for c in color:
        for m in modes:
            for stat, stat_name in zip(stats, stat_names):
                features[stat_name + c + m + 'DeltaScore'] = data['Diff' + c + m + 'Scores'].apply(
                    lambda x: stat(x) if len(x) != 0 else -1)

    def find_advantage(scores):
        for i, x in enumerate(scores):
            if abs(x) >= 100:
                return i
        return 0

    def is_realised_advantage(ind, result, num=0):
        if ind == num and num != 0:
            return 1
        elif ind == 0 and result == 0.5:
            return 1
        elif ind != 0 and ind % 2 != 0 and result == 1:
            return 1
        elif ind != 0 and ind % 2 == 0 and result == 0:
            return 1
        else:
            return 0

    features['FirstAdvantageInd'] = data.Scores.apply(find_advantage)
    features['IsFirstAdvantageRealised'] = features[['FirstAdvantageInd', 'Result']].apply(
        lambda x: is_realised_advantage(*x), axis=1)
    features['FirstAdvantageInd'] = features['FirstAdvantageInd'].apply(lambda x: ceil(x / 2))

    features['LastAdvantageInd'] = data.Scores.apply(lambda x: np.argmax(np.absolute(x) >= 100) or len(x))
    features['IsLastAdvantageRealised'] = features[['LastAdvantageInd', 'Result', 'NumMoves']].apply(
        lambda x: is_realised_advantage(*x), axis=1)
    features['LastAdvantageInd'] = features['LastAdvantageInd'].apply(lambda x: ceil(x / 2))

    def find_erros(x, mode):
        count_erros = 0
        count_blunders = 0
        ind_first_blunder = 0
        ind_first_error = 0
        first_blunder = True
        first_error = True
        for i in range(len(x) - 1):
            sc = abs(x[i + 1] - x[i])
            if sc >= 50:
                count_erros += 1
                if first_error:
                    ind_first_error = i
                    first_error = False
            if sc >= 100:
                count_blunders += 1
                if first_blunder:
                    ind_first_blunder = i
                    first_blunder = False
        if mode == 'ce':
            return count_erros
        elif mode == 'cb':
            return count_blunders
        elif mode == 'ie':
            return ind_first_error
        elif mode == 'ib':
            return ind_first_blunder

    color = ['White', 'Black']
    modes = ['Deb', 'Mit', 'End']

    features['IndError'] = data.Scores.apply(find_erros, args=('ie',))
    features['IndBlunder'] = data.Scores.apply(find_erros, args=('ib',))
    for c in color:
        features['Ind' + c + 'Error'] = data[c + 'Scores'].apply(find_erros, args=('ie',))
        features['Ind' + c + 'Blunder'] = data[c + 'Scores'].apply(find_erros, args=('ib',))

    for m in modes:
        features['Ind' + m + 'Error'] = data[m + 'Scores'].apply(find_erros, args=('ie',))
        features['Ind' + m + 'Blunder'] = data[m + 'Scores'].apply(find_erros, args=('ib',))
        for c in color:
            features['Ind' + c + m + 'Error'] = data[c + m + 'Scores'].apply(find_erros, args=('ie',))
            features['Ind' + c + m + 'Blunder'] = data[c + m + 'Scores'].apply(find_erros, args=('ib',))

    features['NumErrors'] = data.Scores.apply(find_erros, args=('ce',))
    features['NumBlunders'] = data.Scores.apply(find_erros, args=('cb',))
    for c in color:
        features['Num' + c + 'Errors'] = data[c + 'Scores'].apply(find_erros, args=('ce',))
        features['Num' + c + 'Blunders'] = data[c + 'Scores'].apply(find_erros, args=('cb',))

    for m in modes:
        features['Num' + m + 'Errors'] = data[m + 'Scores'].apply(find_erros, args=('ce',))
        features['Num' + m + 'Blunders'] = data[m + 'Scores'].apply(find_erros, args=('cb',))
        for c in color:
            features['Num' + c + m + 'Errors'] = data[c + m + 'Scores'].apply(find_erros, args=('ce',))
            features['Num' + c + m + 'Blunders'] = data[c + m + 'Scores'].apply(find_erros, args=('cb',))

    data['ECO'] = data.Moves.apply(lambda x: x[:6])

    ecos = {}
    cnt = 0
    for moves in data['ECO']:
        str_moves = "".join(map(str, moves))
        if str_moves not in ecos:
            ecos[str_moves] = 1
        else:
            ecos[str_moves] += 1

    def get_eco(x, mode='fr'):
        moves = "".join(map(str, x))
        if ecos[moves] > 50 and mode == 'fr':
            return 1
        elif ecos[moves] <= 50 and ecos[moves] > 10 and mode == 'c':
            return 1
        elif ecos[moves] <= 10 and mode == 'r':
            return 1
        else:
            return 0

    features['FrequentDebut'] = data.ECO.apply(get_eco, args=('fr',))
    features['CommonDebut'] = data.ECO.apply(get_eco, args=('c',))
    features['RareDebut'] = data.ECO.apply(get_eco, args=('r',))

    from chess import QUEEN, PAWN, KNIGHT, BISHOP, ROOK, KING, WHITE, BLACK

    def extract_game_features(uci):
        game = chess.pgn.Game()
        node = game.add_variation(chess.Move.from_uci(uci[0]))
        for move in uci[1:]:
            node = node.add_variation(chess.Move.from_uci(move))

        first_check = True
        first_queen_move = True
        first_king_move = True

        game_features = defaultdict(int)

        count_white_pieces = 37
        count_black_pieces = 37

        node = game
        while node.variations:
            move = node.variation(0).move

            board = node.board()

            moved_piece = board.piece_type_at(move.from_square)
            captured_piece = board.piece_type_at(move.to_square)

            if captured_piece is not None:
                if board.turn == WHITE:
                    count_black_pieces -= captured_piece
                else:
                    count_white_pieces -= captured_piece

            white_advantage = count_white_pieces - count_black_pieces if count_white_pieces > count_black_pieces else 0
            black_advantage = count_black_pieces - count_white_pieces if count_black_pieces > count_white_pieces else 0

            if white_advantage != 0:
                game_features['MovesWhiteAdvantage'] += 1
            if black_advantage != 0:
                game_features['MovesBlackAdvantage'] += 1

            if board.is_kingside_castling(move) and board.turn == WHITE:
                game_features['WhiteKingCastle'] = board.fullmove_number
            elif board.is_queenside_castling(move) and board.turn == WHITE:
                game_features['WhiteQueenCastle'] = board.fullmove_number
            elif board.is_kingside_castling(move) and board.turn == BLACK:
                game_features['BlackKingCastle'] = board.fullmove_number
            elif board.is_queenside_castling(move) and board.turn == BLACK:
                game_features['BlackQueenCastle'] = board.fullmove_number

            if moved_piece == QUEEN and first_queen_move:
                game_features['QueenMoved'] = board.fullmove_number
                first_queen_move = False

            if captured_piece == QUEEN:
                game_features['QuennCaptured'] = board.fullmove_number

            if moved_piece == KING and first_king_move:
                game_features['KingMoved'] = board.fullmove_number
                first_king_move = False

            if move.promotion:
                game_features['Promotion'] += 1
            if board.is_check():
                if board.turn == WHITE:
                    game_features['WhiteСhecks'] += 1
                else:
                    game_features['BlackСhecks'] += 1
                if first_check:
                    game_features['FirstCheck'] = board.fullmove_number
                    first_check = False

            if board.is_en_passant(move):
                game_features['EnPassant'] += 1

            node = node.variation(0)

        if board.is_checkmate():
            game_features['IsCheckmate'] += 1
        if board.is_stalemate():
            game_features['IsStalemate'] += 1
        if board.is_insufficient_material():
            game_features['InsufficientMaterial'] += 1
        if board.can_claim_draw():
            game_features['CanClaimDraw'] += 1

        piece_placement = board.fen().split()[0]
        end_pieces = Counter(x for x in piece_placement if x.isalpha())

        game_features[
            'WhiteAdvantage'] = count_white_pieces - count_black_pieces if count_white_pieces > count_black_pieces else 0
        game_features[
            'BlackAdvantage'] = count_black_pieces - count_white_pieces if count_black_pieces > count_white_pieces else 0

        game_features.update({'End' + piece: cnt
                              for piece, cnt in end_pieces.items()})

        full_game_features = ['MovesWhiteAdvantage', 'MovesBlackAdvantage', 'WhiteKingCastle', 'WhiteQueenCastle',
                              'BlackKingCastle', 'BlackQueenCastle', 'QueenMoved', 'QuennCaptured', 'KingMoved',
                              'Promotion', 'WhiteСhecks', 'BlackСhecks', 'FirstCheck', 'EnPassant',
                              'InsufficientMaterial', 'CanClaimDraw', 'WhiteAdvantage', 'BlackAdvantage',
                              'Endr', 'Endk', 'Endp', 'Endb', 'Endn', 'Endq', 'EndN', 'EndP', 'EndQ', 'EndB', 'EndR',
                              'EndK']

        for f in full_game_features:
            if f not in game_features.keys():
                game_features[f] = 0

        return game_features

    all_features = []
    for uci in data.UCI:
        ext_features = extract_game_features(uci)
        all_features.append(ext_features)

    items = all_features[0].items()
    columns = [i[0] for i in items]
    df_features = pd.DataFrame(columns=columns)

    for ind, feature in enumerate(all_features):
        for key, value in feature.items():
            df_features.loc[ind, key] = value

    features = pd.concat([features, df_features], axis=1, sort=True)
    features = features.fillna(0)

    reject_ser = pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\reject_columns.csv', header=None)
    reject = reject_ser[0].tolist()
    features = features.drop(columns=reject)

    features.columns = [x for x in range(224)]
    reject_ser1 = pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\reject_cols.csv', header=None)
    reject1 = reject_ser1[0].tolist()
    features = features.drop(columns=reject1)

    elos = data.copy()
    cols = elos.columns.tolist()
    cols.remove('WhiteElo')
    cols.remove('BlackElo')
    elos = data.drop(columns=cols)
    elos['MeanElos'] = elos.mean(axis=1).astype(int)
    elos['DiffElos'] = abs(elos.WhiteElo - elos.BlackElo).astype(int)
    elos['SumElos'] = abs(elos.WhiteElo + elos.BlackElo).astype(int)

    with open('kmeans1.pkl', 'rb') as f:
        kmeans1 = pickle.load(f)

    cluster=int(kmeans1.predict(elos)[0])+1

    if cluster==1:
        with open('lr_cl1_mean.pkl', 'rb') as f:
            model_mean = pickle.load(f)
        with open('lr_cl1_diff.pkl', 'rb') as f:
            model_diff = pickle.load(f)

    elif cluster==2:
        with open('lr_cl2_mean.pkl', 'rb') as f:
            model_mean = pickle.load(f)

        with open('lr_cl2_diff.pkl', 'rb') as f:
            model_diff = pickle.load(f)

    elif cluster==3:
        with open('rf_cl3_mean.pkl', 'rb') as f:
            model_mean = pickle.load(f)
        with open('rf_cl3_diff.pkl', 'rb') as f:
            model_diff = pickle.load(f)

    elif cluster==4:
        with open('svm_cl4_mean.pkl', 'rb') as f:
            model_mean = pickle.load(f)
        with open('svm_cl4_diff.pkl', 'rb') as f:
            model_diff = pickle.load(f)
    else:
        return 'Что-то пошло не так...'

    pred_mean =model_mean.predict(features)
    pred_diff = model_diff.predict(features)

    white_rating = pred_mean + pred_diff / 2
    black_rating = pred_mean - pred_diff / 2

    result = 'Рейтинг белых: '+str(int(white_rating[0]))+'\nРейтинг черных: '+str(int(black_rating[0]))
    return result

for i in range(11):
    file='fr'+str(i+1)+'.pgn'
    path='C:\\Users\\Asus\\PredictingRatings\\test_data\\'+file
    if file!='fr4.pgn':
        names=get_rating(path, 'names')
        print(names)
        current = get_rating(path, 'current')
        print(current)
        ratings = get_rating(path, 'ratings')
        print(ratings)
    print('\n\n')
