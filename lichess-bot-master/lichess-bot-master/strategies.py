"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult, Limit
import random
from engine_wrapper import EngineWrapper
from config import Configuration
from typing import Optional, Any, Dict, List
# from engine_libs import *
OPTIONS_TYPE = Dict[str, Any]
COMMANDS_TYPE = List[str]


class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """
    def __init__(self, main_engine: MinimalEngine, name: Optional[str] = None) -> None:
        self.id = {
            "name": name
        }
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name: str) -> Any:
        main_engine = self.main_engine

        def method(*args: Any, **kwargs: Dict[str, Any]) -> Any:
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """
    def __init__(self, commands: COMMANDS_TYPE, options: OPTIONS_TYPE, stderr: Optional[int],
                 draw_or_resign: Configuration, name: Optional[str] = None, **popen_args: Dict[str, str]) -> None:
        super().__init__(options, draw_or_resign)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {
            "name": self.engine_name
        }
        self.initialize_model()

    def get_pid(self) -> str:
        return "?"

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool,
               root_moves: List[chess.Move]) -> None:
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self>
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass

    def initialize_model(self):
        raise NotImplementedError(('Intialize the model'))




class ExampleEngine(MinimalEngine):
    pass

import pickle as pkl
#import load_model
from tensorflow.keras.models import load_model
import chess
import numpy as np
# load model

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

def check_turn(board_state):
    if board_state.split(' ')[1] == 'w':
        return 'w'
    else:
        return 'b'
    

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



# Strategy names and ideas from tom7's excellent eloWorld video
class RandomForestKasparv(ExampleEngine):

    def initialize_model(self):
        print('loading model')
        self.model = load_model('../../chessengine.h5')

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool,
               root_moves: List[chess.Move],*args: Any) -> PlayResult:
        # print(root_moves, draw_offered, time_limit, board)
        # if len(board.move_stack) == 0:
        #     return PlayResult(chess.Move.from_uci('e2e4'), None)
        
        legal_moves = [move for move in board.legal_moves]
        best_score = float('-inf')
        best_move_index = None

        for i, move in enumerate(legal_moves):
            board.push(move)
            x = np.array([label_helper(board.fen()).flatten()])
            y = self.model.predict(x)[0]
            score = y.max()  # get the highest score out of all classes
            if score > best_score:
                best_score = score
                best_move_index = i
            board.pop()
            # print(move, score)
        
        best_move = legal_moves[best_move_index]
        print(best_move, best_score)
        print(type(best_move))



        # next_move = self.model.make_prediction(board.move_stack[-1].uci(), board)

        return PlayResult(best_move, None)

class RandomMove(ExampleEngine):
    def initialize_model(self):
        print('model initalization')
        pass
    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool,
               root_moves: List[chess.Move],*args: Any) -> PlayResult:
        # print(root_moves, draw_offered, time_limit, board)
        print(board.move_stack)
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)
