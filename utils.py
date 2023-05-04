def get_moves(game):
    moves = []
    for i in game.mainline_moves():
        moves.append(i)
    return moves

import numpy as np
import chess

def get_piece_type(piece):
    # Helper function to get the type of a chess piece (with color information)
    if piece is None:
        return None
    elif piece.color == chess.WHITE:
        if piece.piece_type == chess.PAWN:
            return 'P'
        elif piece.piece_type == chess.KNIGHT:
            return 'N'
        elif piece.piece_type == chess.BISHOP:
            return 'B'
        elif piece.piece_type == chess.ROOK:
            return 'R'
        elif piece.piece_type == chess.QUEEN:
            return 'Q'
        elif piece.piece_type == chess.KING:
            return 'K'
    elif piece.color == chess.BLACK:
        if piece.piece_type == chess.PAWN:
            return 'p'
        elif piece.piece_type == chess.KNIGHT:
            return 'n'
        elif piece.piece_type == chess.BISHOP:
            return 'b'
        elif piece.piece_type == chess.ROOK:
            return 'r'
        elif piece.piece_type == chess.QUEEN:
            return 'q'
        elif piece.piece_type == chess.KING:
            return 'k'

def get_board_matrix(board_state):
    # Initialize an 8x8 matrix to represent the board
    matrix = np.zeros((8, 8), dtype=str)

    # Parse the board_state string to obtain the positions of the pieces on the board
    board = chess.Board(board_state)
    for row in range(8):
        for col in range(8):
            # Get the square index corresponding to the current row and column
            square = chess.square(col, 7 - row)

            # Get the type of the piece occupying the current square (if any)
            piece_type = get_piece_type(board.piece_at(square))

            # Store the piece type in the matrix
            matrix[row][col] = piece_type or '.'

    return matrix

le = preprocessing.LabelEncoder()

def label_encode_game(game):
    board = game.board()
    moves = get_moves(game)
    encoded_moves = []
    for move in moves:
        board.push(move)
        encoded_moves.append(le.fit_transform(get_board_matrix(board.fen()).flatten()))
    return encoded_moves
def get_node(game):
    comments = []
    for node in game.mainline_moves():
        comments.append(node.comment)
    return comments
        
def get_evaluation(game):
    evaluation = []
    for node in game.mainline():
        comment = node.comment
        comment = comment.split('/')
        evaluation.append(comment[0])
    return evaluation
def check_turn(board_state):
    if board_state.split(' ')[1] == 'w':
        return 'w'
    else:
        return 'b' 
def label_encode_for_white(board_matrix):

    new_board_matrix = np.zeros((8, 8), dtype=int)
   
    for i in range(len(board_matrix)):
        for j in range(len(board_matrix[i])):
            if board_matrix[i][j] == 'P':
                new_board_matrix[i][j] = 1
            elif board_matrix[i][j] == 'N':
                new_board_matrix[i][j] = 3
            elif board_matrix[i][j] == 'B':
                new_board_matrix[i][j] = 3
            elif board_matrix[i][j] == 'R':
                new_board_matrix[i][j] = 5
            elif board_matrix[i][j] == 'Q':
                new_board_matrix[i][j] = 9
            elif board_matrix[i][j] == 'K':
                new_board_matrix[i][j] = 10
            elif board_matrix[i][j] == '.':
                new_board_matrix[i][j] = 0
            elif board_matrix[i][j] == 'p':
                new_board_matrix[i][j] = -1
            elif board_matrix[i][j] == 'n':
                new_board_matrix[i][j] = -3
            elif board_matrix[i][j] == 'b':
                new_board_matrix[i][j] = -3
            elif board_matrix[i][j] == 'r':
                new_board_matrix[i][j] = -5
            elif board_matrix[i][j] == 'q':
                new_board_matrix[i][j] = -9
            elif board_matrix[i][j] == 'k':
                new_board_matrix[i][j] = -10
    return new_board_matrix


def label_encode_for_black(board_matrix):
    new_board_matrix = np.zeros((8, 8), dtype=int)
    
    for i in range(len(board_matrix)):
        for j in range(len(board_matrix[i])):
            if board_matrix[i][j] == 'P':
                new_board_matrix[i][j] = -1
            elif board_matrix[i][j] == 'N':
                new_board_matrix[i][j] = -3
            elif board_matrix[i][j] == 'B':
                new_board_matrix[i][j] = -3
            elif board_matrix[i][j] == 'R':
                new_board_matrix[i][j] = -5
            elif board_matrix[i][j] == 'Q':
                new_board_matrix[i][j] = -9
            elif board_matrix[i][j] == 'K':
                new_board_matrix[i][j] = -10
            elif board_matrix[i][j] == '.':
                new_board_matrix[i][j] = 0
            elif board_matrix[i][j] == 'p':
                new_board_matrix[i][j] = 1
            elif board_matrix[i][j] == 'n':
                new_board_matrix[i][j] = 3
            elif board_matrix[i][j] == 'b':
                new_board_matrix[i][j] = 3
            elif board_matrix[i][j] == 'r':
                new_board_matrix[i][j] = 5
            elif board_matrix[i][j] == 'q':
                new_board_matrix[i][j] = 9
            elif board_matrix[i][j] == 'k':
                new_board_matrix[i][j] = 10
    return new_board_matrix

def label_helper(board_state):
    board_matrix = get_board_matrix(board_state)
    if check_turn(board_state) == 'w':
       board_matrix = label_encode_for_black(board_matrix)
    elif check_turn(board_state) == 'b':
       board_matrix = label_encode_for_white(board_matrix)
    return board_matrix
        
# new label encoding
def label_encode_game_based_on_turn(game):
    board = game.board()
    moves = get_moves(game)
    encoded_moves = []
    for move in moves:
        board.push(move)
        board_state = board.fen()
        # encoded_moves.append(le.fit_transform(get_board_matrix(board.fen()).flatten()))
        encoded_moves.append(label_helper(board_state).flatten())
    return encoded_moves