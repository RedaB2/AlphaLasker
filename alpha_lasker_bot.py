import time
import random
from typing import List, Tuple, Optional

# Constants
time_limit = 5  # Move time limit in seconds
stalemate_threshold = 20  # Moves without mills forming before a draw

class LaskerMorris:
    def __init__(self):
        self.board = self.initialize_board()
        self.hands = {'blue': 10, 'orange': 10}  # Stones in hand
        self.turn = 'blue'  # Blue starts
        self.move_count = 0
        self.no_mill_moves = 0
    
    def initialize_board(self):
        return {point: None for point in self.valid_points()}
    
    def valid_points(self):
       return [
            "a7", "d7", "g7", "b6", "d6", "f6", "c5", "d5", "e5", 
            "a4", "b4", "c4", "e4", "f4", "g4", "c3", "d3", "e3", 
            "b2", "d2", "f2", "a1", "d1", "g1"
        ]

    def generate_moves(self, player: str) -> List[Tuple[str, str, str]]:
        moves = []
        if self.hands[player] > 0:
            for point in self.board:
                if self.board[point] is None:
                    moves.append((f"h{1 if player == 'blue' else 2}", point, "r0"))
        return moves
    
    def make_move(self, move: Tuple[str, str, str]):
        src, dest, remove = move
        if src.startswith("h"):
            self.board[dest] = self.turn
            self.hands[self.turn] -= 1
        if remove != "r0":
            self.board[remove] = None
        self.turn = "orange" if self.turn == "blue" else "blue"
        self.move_count += 1
    
    def undo_move(self, move: Tuple[str, str, str]):
        src, dest, remove = move
        self.board[dest] = None
        if src.startswith("h"):
            self.hands[self.turn] += 1
        if remove != "r0":
            self.board[remove] = self.turn
        self.turn = "orange" if self.turn == "blue" else "blue"
        self.move_count -= 1
    
    def evaluate(self) -> int:
        return sum(1 for v in self.board.values() if v == "blue") - sum(1 for v in self.board.values() if v == "orange")
    
    def minimax(self, depth: int, alpha: int, beta: int, maximizing: bool) -> Tuple[int, Optional[Tuple[str, str, str]]]:
        if depth == 0:
            return self.evaluate(), None
        
        best_move = None
        moves = self.generate_moves(self.turn if maximizing else ("orange" if self.turn == "blue" else "blue"))
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                self.make_move(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, False)
                self.undo_move(move)
                if eval > max_eval:
                    max_eval, best_move = eval, move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                self.make_move(move)
                eval, _ = self.minimax(depth - 1, alpha, beta, True)
                self.undo_move(move)
                if eval < min_eval:
                    min_eval, best_move = eval, move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def best_move(self, max_depth=3):
        start_time = time.time()
        _, move = self.minimax(max_depth, float('-inf'), float('inf'), True)
        return move if time.time() - start_time < time_limit else None
    
    def play(self):
        while True:
            move = self.best_move()
            if move is None:
                print(f"{self.turn} loses: timeout or no valid move.")
                break
            print(f"{self.turn} plays: {move}")
            self.make_move(move)
            if self.move_count > stalemate_threshold:
                print("Game ends in stalemate.")
                break

if __name__ == "__main__":
    game = LaskerMorris()
    game.play()
