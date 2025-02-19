import time
import random
from typing import List, Tuple, Optional

# Constants
TIME_LIMIT = 5  #time limit for ai
STALEMATE_THRESHOLD = 20  #moves without mill draw
VALID_POINTS = [
    "a7", "d7", "g7", "b6", "d6", "f6", "c5", "d5", "e5",
    "a4", "b4", "c4", "e4", "f4", "g4", "c3", "d3", "e3",
    "b2", "d2", "f2", "a1", "d1", "g1"
]

# Game Logic
class LaskerMorris:
    def __init__(self):
        self.board = self.initialize_board()
        self.hands = {'blue': 10, 'orange': 10}  #stones in hand
        self.turn = None  #will be assigned by the referee
        self.move_count = 0
        self.no_mill_moves = 0

    def initialize_board(self):
        return {point: None for point in VALID_POINTS}

    def generate_moves(self, player: str) -> List[Tuple[str, str, str]]:
        moves = []
        if self.hands[player] > 0:

            #2:phase 1:placing stones
            for point in self.board:
                if self.board[point] is None:
                    moves.append((f"h{1 if player == 'blue' else 2}", point, "r0"))
        else:
            #2: phase:moving stones
            for src in self.board:
                if self.board[src] == player:
                    for dest in self.get_adjacent_points(src):
                        if self.board[dest] is None:
                            moves.append((src, dest, "r0"))
        return moves

    def get_adjacent_points(self, point: str) -> List[str]:
        #adjacent points for each board point
        adjacency = {
            "a7": ["d7", "a4"],
            "d7": ["a7", "g7", "d6"],
            "g7": ["d7", "g4"],
            "b6": ["d6", "b4"],
            "d6": ["b6", "d7", "f6", "d5"],
            "f6": ["d6", "f4"],
            "c5": ["d5", "c4"],
            "d5": ["c5", "d6", "e5", "d4"],
            "e5": ["d5", "e4"],
            "a4": ["a7", "b4", "a1"],
            "b4": ["a4", "b6", "c4", "b2"],
            "c4": ["b4", "c5", "c3"],
            "e4": ["e5", "f4", "e3"],
            "f4": ["e4", "f6", "g4", "f2"],
            "g4": ["f4", "g7", "g1"],
            "c3": ["c4", "d3"],
            "d3": ["c3", "d2", "e3", "d1"],
            "e3": ["d3", "e4"],
            "b2": ["b4", "d2"],
            "d2": ["b2", "d3", "f2"],
            "f2": ["d2", "f4"],
            "a1": ["a4", "d1"],
            "d1": ["a1", "d3", "g1"],
            "g1": ["d1", "g4"]
        }
        return adjacency.get(point, [])

    def make_move(self, move: Tuple[str, str, str]):
        src, dest, remove = move
        if src.startswith("h"):
            #place a stone from hand
            self.board[dest] = self.turn
            self.hands[self.turn] -= 1
        else:
            #move a stone on the board
            self.board[dest] = self.board[src]
            self.board[src] = None
        if remove != "r0":
            #remove opponent's stone
            self.board[remove] = None
        self.turn = "orange" if self.turn == "blue" else "blue"
        self.move_count += 1

    def is_mill_formed(self, move: Tuple[str, str, str]) -> bool:
        #check if the move forms a mill
        _, dest, _ = move
        color = self.board[dest]
        if color is None:
            return False

        #check horizontal and vertical mills
        lines = [
            ["a7", "d7", "g7"], ["b6", "d6", "f6"], ["c5", "d5", "e5"],
            ["a4", "b4", "c4"], ["e4", "f4", "g4"], ["c3", "d3", "e3"],
            ["b2", "d2", "f2"], ["a1", "d1", "g1"]
        ]
        for line in lines:
            if dest in line and all(self.board[point] == color for point in line):
                return True
        return False

    def print_board(self):
        #print the current state of the board
        print("\nCurrent Board State:")
        for point in VALID_POINTS:
            stone = self.board[point]
            if stone is None:
                print(f"{point}: Empty", end=" | ")
            else:
                print(f"{point}: {stone}", end=" | ")
        print("\n")

#player logic
class Player:
    def __init__(self, color):
        self.color = color

    def make_move(self, game: LaskerMorris) -> Tuple[str, str, str]:
        #generate all possible moves and pick one randomly
        moves = game.generate_moves(self.color)
        if not moves:
            raise Exception(f"{self.color} has no valid moves left!")
        return random.choice(moves)

#human player logic
class HumanPlayer:
    def __init__(self, color):
        self.color = color

    def make_move(self, game: LaskerMorris) -> Tuple[str, str, str]:
        #human player inputs their move
        print(f"\nYour turn ({self.color}). Enter your move:")
        print("Format: <source> <destination> <remove> (e.g., h1 a7 r0)")
        move = input("Your move: ").strip().split()
        return tuple(move)

#referee logic
class Referee:
    def __init__(self, player1, player2):
        self.players = {'blue': player1, 'orange': player2}
        self.current_player = None  #will be assigned randomly
        self.game = LaskerMorris()
        self.time_limit = TIME_LIMIT
        self.stalemate_threshold = STALEMATE_THRESHOLD
        self.move_count = 0
        self.no_mill_moves = 0

    def assign_colors(self):
        #randomly assign colors to players
        colors = ['blue', 'orange']
        random.shuffle(colors)
        self.players['blue'].color = colors[0]
        self.players['orange'].color = colors[1]

        #randomly decide who goes first
        self.current_player = random.choice(['blue', 'orange'])
        self.game.turn = self.current_player  #update the turn in the game class
        print(f"\nPlayer 1 is {colors[0]}, Player 2 is {colors[1]}")
        print(f"{self.current_player} will go first.")

    def start_game(self):
        self.assign_colors()
        self.game.print_board()
        while not self.is_game_over():
            self.play_turn()
        self.declare_winner()

    def play_turn(self):
        player = self.players[self.current_player]
        print(f"\n{self.current_player}'s turn:")

        if isinstance(player, HumanPlayer):
            #human turn no time limit
            move = player.make_move(self.game)
        else:
            #ai turn with time limit
            start_time = time.time()
            try:
                move = player.make_move(self.game)
            except Exception as e:
                print(f"{self.current_player} loses: {str(e)}")
                self.end_game(self.current_player, str(e))
                return

            elapsed_time = time.time() - start_time
            if elapsed_time > self.time_limit:
                print(f"{self.current_player} loses: timeout.")
                self.end_game(self.current_player, "Time out!")
                return

        if not self.is_valid_move(move):
            print(f"{self.current_player} loses: invalid move.")
            self.end_game(self.current_player, f"Invalid move {move}!")
            return

        self.game.make_move(move)
        self.move_count += 1

        if self.game.is_mill_formed(move):
            self.no_mill_moves = 0
        else:
            self.no_mill_moves += 1

        print(f"{self.current_player} plays: {move}")
        self.game.print_board()
        self.current_player = 'orange' if self.current_player == 'blue' else 'blue'

    def is_valid_move(self, move: Tuple[str, str, str]) -> bool:
        #check if the move is valid
        src, dest, remove = move
        if dest not in VALID_POINTS:
            return False
        if src.startswith("h"):
            if self.game.hands[self.current_player] <= 0:
                return False
        else:
            if self.game.board[src] != self.current_player:
                return False
        if self.game.board[dest] is not None:
            return False
        if remove != "r0" and self.game.board[remove] is None:
            return False
        return True

    def is_game_over(self) -> bool:
        #check if the game has reached a terminal state
        if self.move_count >= self.stalemate_threshold and self.no_mill_moves >= self.stalemate_threshold:
            return True
        if self.game.hands['blue'] == 0 and self.game.hands['orange'] == 0:
            return True
        return False

    def end_game(self, loser: str, reason: str):
        winner = 'orange' if loser == 'blue' else 'blue'
        print(f"\nEND: {winner} WINS! {loser} LOSES! {reason}")
        exit()

    def declare_winner(self):
        #below determine the winner based on the game state
        if self.move_count >= self.stalemate_threshold and self.no_mill_moves >= self.stalemate_threshold:
            print("\nDraw!")
        else:
            if self.game.hands['blue'] == 0:
                print("\nEND: orange WINS! blue LOSES! Ran out of pieces!")
            else:
                print("\nEND: blue WINS! orange LOSES! Ran out of pieces!")

# Main Execution
if __name__ == "__main__":
    print("Welcome to Lasker Morris!")
    print("You will play against the AI.")
    human_player = HumanPlayer(None)  #color will be assigned by ref
    ai_player = Player(None)  #color will be assigned by the ref
    referee = Referee(human_player, ai_player)
    referee.start_game()