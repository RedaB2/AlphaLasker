import time
import sys
from typing import List, Tuple, Dict, Optional

# Constants
time_limit = 5  # move time limit in seconds
stalemate_threshold = 20  # moves without mills forming before a draw

class LaskerMorris:
    def __init__(self):

        self.board: Dict[str, Optional[str]] = self.initialize_board()

        self.hands = {'blue': 10, 'orange': 10}

        self.turn = 'blue' 

        self.no_mill_moves = 0
        
        self.move_count = 0 # not necessary can comment if you want

        # waiting for the colors to be assigned by referee
        self.my_color: Optional[str] = None
        self.opp_color: Optional[str] = None

        self.adjacency_list = self.build_adjacency_list() # for each valid points

        self.mill_combinations = self.build_mill_combinations() # all possible mills first

    def initialize_board(self) -> Dict[str, Optional[str]]:
        """create a dictionary with all valid points set to None initially."""
        board = {}
        for pt in self.valid_points():
            board[pt] = None
        return board

    def valid_points(self) -> List[str]:
        """all valid Lasker Morris positions."""
        return [
            "a7", "d7", "g7", "b6", "d6", "f6", "c5", "d5", "e5", 
            "a4", "b4", "c4", "e4", "f4", "g4", "c3", "d3", "e3", 
            "b2", "d2", "f2", "a1", "d1", "g1"
        ]

    def build_adjacency_list(self) -> Dict[str, List[str]]:
        adjacency = {
            "a7": ["d7", "a4"],
            "d7": ["a7", "g7", "d6"],
            "g7": ["d7", "g4"],
            "b6": ["d6", "b4"],
            "d6": ["b6", "f6", "d5", "d7"],
            "f6": ["d6", "f4"],
            "c5": ["d5", "c4"],
            "d5": ["c5", "e5", "d6"],
            "e5": ["d5", "e4"],
            "a4": ["a7", "b4", "a1"],
            "b4": ["a4", "c4", "b6"],
            "c4": ["b4", "c3", "c5"],
            "e4": ["e5", "f4", "e3"],
            "f4": ["e4", "g4", "f6"],
            "g4": ["f4", "g7", "g1"],
            "c3": ["c4", "d3"],
            "d3": ["c3", "e3", "d2"],
            "e3": ["d3", "e4"],
            "b2": ["d2", "b4"],
            "d2": ["b2", "f2", "d3"],
            "f2": ["d2", "f4"],
            "a1": ["a4", "d1"],
            "d1": ["a1", "g1", "d2"],
            "g1": ["d1", "g4"]
        }
        return adjacency

    def build_mill_combinations(self) -> List[Tuple[str, str, str]]:

        possible_mills = [
            ("a7","d7","g7"),
            ("b6","d6","f6"),
            ("c5","d5","e5"),
            ("a4","b4","c4"),
            ("e4","f4","g4"),
            ("c3","d3","e3"),
            ("b2","d2","f2"),
            ("a1","d1","g1"),
            # vertical mills
            ("a7","a4","a1"),
            ("b6","b4","b2"),
            ("c5","c4","c3"),
            ("d7","d6","d5"),
            ("d3","d2","d1"),
            ("e5","e4","e3"),
            ("f6","f4","f2"),
            ("g7","g4","g1")
        ]
        # sort each triple for easy search
        sorted_mills = []
        for trip in possible_mills:
            sorted_mills.append(tuple(sorted(trip)))
        return sorted_mills

    def current_player_stones_on_board(self, player: str) -> List[str]:
        """Return a list of board points where the current player has a stone."""
        return [pt for pt, occupant in self.board.items() if occupant == player]

    def opponent(self, player: str) -> str:
        return 'blue' if player == 'orange' else 'orange'

    def get_phase(self, player: str) -> str:

        stones_on_board = len(self.current_player_stones_on_board(player))
        if self.hands[player] > 0:
            return "placing"
        elif stones_on_board > 3:
            return "moving"
        else:

            return "flying"

    def forms_mill(self, move_point: str, player: str) -> bool:
        for trip in self.mill_combinations:
            if move_point in trip:

                if all((pt == move_point or self.board[pt] == player) for pt in trip):
                    return True
        return False

    def all_mills_of_player(self, player: str) -> List[Tuple[str, str, str]]:
        """
        return all mills on the current board for the given player.
        """
        result = []
        for trip in self.mill_combinations:
            if all(self.board[pt] == player for pt in trip):
                result.append(trip)
        return result

    def stone_is_in_mill(self, pt: str, player: str) -> bool:
        for trip in self.mill_combinations:
            if pt in trip:
                if all(self.board[x] == player for x in trip):
                    return True
        return False

    def generate_moves(self, player: str) -> List[Tuple[str, str, str]]:
        """
        generate all valid moves (A, B, C):
          A = source of stone ("h1"/"h2" or board point)
          B = destination on board
          C = opponent stone to remove or "r0"
        """
        moves = []
        opponent = self.opponent(player)
        phase = self.get_phase(player)

        # 1) if in "placing" phase, we can place from hand to any empty point
        if phase == "placing":
            if self.hands[player] > 0:
                for dest in self.valid_points():
                    if self.board[dest] is None:
                        # mill?
                        # what if stone placed?
                        if self.forms_mill(dest, player):
    
                            remove_candidates = self.get_remove_candidates(opponent)
                            for r in remove_candidates:
                                moves.append((f"h{1 if player=='blue' else 2}", dest, r))
                        else:
                            # no mill formed => no removal
                            moves.append((f"h{1 if player=='blue' else 2}", dest, "r0"))

        else:

            stones = self.current_player_stones_on_board(player)
            for src in stones:
                if phase == "moving":
               
                    destinations = [p for p in self.adjacency_list[src] if self.board[p] is None]
                else:
                
                    destinations = [p for p in self.board if self.board[p] is None]

                for dest in destinations:
                    
                    if self.forms_mill_after_move(src, dest, player):
                    
                        remove_candidates = self.get_remove_candidates(opponent)
                        for r in remove_candidates:
                            moves.append((src, dest, r))
                    else:
                  
                        moves.append((src, dest, "r0"))
        return moves

    def get_remove_candidates(self, opponent: str) -> List[str]:
        opp_stones = self.current_player_stones_on_board(opponent)
       
        not_in_mill = [pt for pt in opp_stones if not self.stone_is_in_mill(pt, opponent)]
        if len(not_in_mill) > 0:
            return not_in_mill
        else:
            return opp_stones

    def forms_mill_after_move(self, src: str, dest: str, player: str) -> bool:
        orig_src_occupant = self.board[src]
        self.board[src] = None
        # dest forms a mill
        mill_formed = self.forms_mill(dest, player)
        # revert
        self.board[src] = orig_src_occupant
        return mill_formed

    def make_move(self, move: Tuple[str, str, str]):
        src, dest, remove = move

    
        if src.startswith("h"):
            
            self.hands[self.turn] -= 1
            self.board[dest] = self.turn
        else:
            self.board[dest] = self.turn
            self.board[src] = None

        # if needed we rm the opponent stone
        if remove != "r0":
            self.board[remove] = None

        self.move_count += 1

        if remove != "r0":
            self.no_mill_moves = 0
        else:
            self.no_mill_moves += 1

        self.turn = self.opponent(self.turn)

    def undo_move(self, move: Tuple[str, str, str], prev_turn: str):
        src, dest, remove = move

        # turn back
        self.turn = prev_turn
        
        # from hand
        if src.startswith("h"):
            self.hands[prev_turn] += 1
            self.board[dest] = None
        else:
            # src
            self.board[src] = prev_turn
            self.board[dest] = None

        if remove != "r0":
            opp = self.opponent(prev_turn)
            self.board[remove] = opp
        self.move_count -= 1

    def is_terminal_state(self) -> bool:
        """
        check for board state
        """
        blue_total = len(self.current_player_stones_on_board("blue")) + self.hands["blue"]
        orange_total = len(self.current_player_stones_on_board("orange")) + self.hands["orange"]
        if blue_total <= 2 or orange_total <= 2:
            return True

        if not self.generate_moves(self.turn):
            return True

        if self.no_mill_moves >= stalemate_threshold:
            return True

        return False

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal_state():
            return None

        # stalemate first?
        if self.no_mill_moves >= stalemate_threshold:
            return 'draw'

        # if a player has <= 2 stones in total
        blue_total = len(self.current_player_stones_on_board("blue")) + self.hands["blue"]
        orange_total = len(self.current_player_stones_on_board("orange")) + self.hands["orange"]
        if blue_total <= 2 and orange_total > 2:
            return "orange"
        if orange_total <= 2 and blue_total > 2:
            return "blue"

        # curr player can't do any move
        if not self.generate_moves(self.turn):
            # the current player is the loser, so the other wins
            return self.opponent(self.turn)

        # none of above, so we draw
        return 'draw'

    # -------------------- EVAL -------------------- #

    def evaluate(self, player: str) -> int:
        p_stones = len(self.current_player_stones_on_board(player))
        o_stones = len(self.current_player_stones_on_board(self.opponent(player)))
        # diff
        score = (p_stones - o_stones) * 10

        p_mills = len(self.all_mills_of_player(player))
        o_mills = len(self.all_mills_of_player(self.opponent(player)))
        score += (p_mills - o_mills) * 2

        return score

    def minimax(self,
                depth: int,
                alpha: float,
                beta: float,
                maximizing_player: bool,
                base_player: str) -> Tuple[float, Optional[Tuple[str, str, str]]]:
        """
        depth: how deep we search
        alpha, beta: for alpha-beta pruning
        maximizing_player: true if the node is for 'base_player' to move
        base_player: the player from whose perspective we are evaluating
        """
        if depth == 0 or self.is_terminal_state():
            return self.evaluate(base_player), None

        current_player = self.turn
        moves = self.generate_moves(current_player)
        if not moves:
            # no moves => terminal from our perspective
            return self.evaluate(base_player), None

        bestial_move = None

        if maximizing_player:
            value = float('-inf')
            for move in moves:
                self.make_move(move)
                score, _ = self.minimax(depth - 1, alpha, beta,
                                        not maximizing_player,
                                        base_player)
                self.undo_move(move, current_player)
                if score > value:
                    value = score
                    bestial_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, bestial_move
        else:
            value = float('inf')
            for move in moves:
                self.make_move(move)
                score, _ = self.minimax(depth - 1, alpha, beta,
                                        not maximizing_player,
                                        base_player)
                self.undo_move(move, current_player)
                if score < value:
                    value = score
                    bestial_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, bestial_move
        

    # too slow so we can change depth to 6 isntead of 3
    def new_bestial_move(self, max_depth: int = 6) -> Optional[Tuple[str, str, str]]:
        """
        Iterative deepening approach:
          - Start from depth=1 up to max_depth
          - Stop if time is exhausted.
          - Keep track of the best move found so far.
        """
        start = time.time()
        best = None
        # We'll search from 1..max_depth
        for depth in range(1, max_depth + 1):
            # Check if time is already nearly out
            if (time.time() - start) > time_limit - 0.05:
                break
            val, move = self.minimax(depth, float('-inf'), float('inf'),
                                     True, self.turn, start)
            # If we still have time left, update best move
            if (time.time() - start) <= time_limit - 0.05:
                best = move
            else:
                # no time left
                break
        return best

    # local testing
    def play(self):
        while True:
            if self.is_terminal_state():
                winner = self.get_winner()
                if winner == 'draw':
                    print("Game ends in a draw.")
                else:
                    print(f"Game Over. Winner is {winner}.")
                break

            move = self.new_bestial_move()
            if move is None:
                # no moves other player wins
                print(f"{self.turn} cannot move or times out. {self.opponent(self.turn)} wins!")
                break

            print(f"{self.turn} plays: {move}")
            self.make_move(move)

            # no infinite loops
            if self.move_count > 200:
                print("Forcing stop after 200 moves.")
                break

    # ref communication

    def run(self):
        """
        1) wait for color assignment ("blue" or "orange").
        2) if we're "blue", we move first.
        3) otherwise, read the first move from the "blue" player, apply it, then move.
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

        
            if self.my_color is None:
                if line in ["blue", "orange"]:
                    self.my_color = line
                    self.opp_color = self.opponent(line)
                
                    if self.my_color == 'blue':
                        # Our turn is also 'blue'
                        self.turn = 'blue'
                        self.do_our_move()
                    else:
                        
                        self.turn = 'blue'
                    continue
                else:
                    
                    continue

            # check for the END signal
            if line.startswith("END:"):
    
                break

        
            move_parts = line.split()
            if len(move_parts) != 3:
                
                print(f"INVALID MOVE {line}", flush=True)
                break

            src, dest, remove = move_parts

            
            if self.turn != self.my_color:
                
                valid_opponent_moves = self.generate_moves(self.turn)
                if (src, dest, remove) not in valid_opponent_moves:
                   
                    print(f"INVALID MOVE {src} {dest} {remove}", flush=True)
                    break
                else:
                    
                    self.make_move((src, dest, remove))
            else:
                
                valid_our_moves = self.generate_moves(self.turn)
                if (src, dest, remove) not in valid_our_moves:
                    print(f"INVALID MOVE {src} {dest} {remove}", flush=True)
                    break
                else:
                    self.make_move((src, dest, remove))

            if not self.is_terminal_state() and self.turn == self.my_color:
                self.do_our_move()

    def do_our_move(self):
        """
        helper to compute best move and flush
        """
        move = self.new_bestial_move()
        if move is None:
    
            print(f"h{1 if self.my_color=='blue' else 2} X0 r0", flush=True)
            return
        src, dest, remove = move
        print(f"{src} {dest} {remove}", flush=True)
        self.make_move(move)

if __name__ == "__main__":
    game = LaskerMorris()
    # uncomment for local testing no ref package
    # game.play()
    game.run()