import sys
import time
from typing import List, Tuple, Dict, Optional


time_limit = 5  # move time limit in seconds
stalemate_threshold = 20  # moves without mills forming before a draw

class LaskerMorris:
    """
    Implements the game logic and AI for Lasker Morris.
    This class handles board management, move generation, game state evaluation, and interaction with the referee.
    """
    def __init__(self):
        """
        Initializes the game board, player states, adjacency lists, and mill combinations.
        """
        self.board: Dict[str, Optional[str]] = self.initialize_board()

        self.hands = {'blue': 10, 'orange': 10}
        self.turn = 'blue'
        self.no_mill_moves = 0
        self.move_count = 0  # tracks total moves made

        # Assigned at start by reading referee's first line:
        self.my_color: Optional[str] = None
        self.opp_color: Optional[str] = None

        # Build adjacency list
        self.adjacency_list = self.build_adjacency_list()
        # All possible mill combinations
        self.mill_combinations = self.build_mill_combinations()

    def initialize_board(self) -> Dict[str, Optional[str]]:
        """
        Creates an empty board where all valid positions are set to None.
        """
        board = {}
        for pt in self.valid_points():
            board[pt] = None
        return board

    def valid_points(self) -> List[str]:
        """
        Returns a list of valid board positions.
        """
        return [
            "a7", "d7", "g7", "b6", "d6", "f6", 
            "c5", "d5", "e5", "a4", "b4", "c4", 
            "e4", "f4", "g4", "c3", "d3", "e3", 
            "b2", "d2", "f2", "a1", "d1", "g1"
        ]

    def build_adjacency_list(self) -> Dict[str, List[str]]:
        """
        Creates an adjacency list for valid board positions.
        """
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
            "b4": ["a4", "c4", "b6", "b2"],
            "c4": ["b4", "c3", "c5"],
            "e4": ["e5", "f4", "e3"],
            "f4": ["e4", "g4", "f6", "f2"],
            "g4": ["f4", "g7", "g1"],
            "c3": ["c4", "d3"],
            "d3": ["c3", "e3", "d2"],
            "e3": ["d3", "e4"],
            "b2": ["d2", "b4"],
            "d2": ["b2", "f2", "d3", "d1"],  # slight adjacency fix
            "f2": ["d2", "f4"],
            "a1": ["a4", "d1"],
            "d1": ["a1", "g1", "d2"],
            "g1": ["d1", "g4"]
        }
        return adjacency

    def build_mill_combinations(self) -> List[Tuple[str, str, str]]:
        """
        Returns all possible mill formations on the board.
        """
        possible_mills = [
            # horizontal mills
            ("a7", "d7", "g7"),
            ("b6", "d6", "f6"),
            ("c5", "d5", "e5"),
            ("a4", "b4", "c4"),
            ("e4", "f4", "g4"),
            ("c3", "d3", "e3"),
            ("b2", "d2", "f2"),
            ("a1", "d1", "g1"),
            # vertical mills
            ("a7", "a4", "a1"),
            ("b6", "b4", "b2"),
            ("c5", "c4", "c3"),
            ("d7", "d6", "d5"),
            ("d3", "d2", "d1"),
            ("e5", "e4", "e3"),
            ("f6", "f4", "f2"),
            ("g7", "g4", "g1")
        ]
        # Sort each triple for easier checking
        sorted_mills = []
        for trip in possible_mills:
            sorted_mills.append(tuple(sorted(trip)))
        return sorted_mills

    def current_player_stones_on_board(self, player: str) -> List[str]:
        """
        Returns a list of board positions occupied by the given player.
        """
        return [pt for pt, occupant in self.board.items() if occupant == player]

    def opponent(self, player: str) -> str:
        """
        Returns the opponent's color.
        """
        return 'blue' if player == 'orange' else 'orange'

    def get_phase(self, player: str) -> str:
        """
        Determines the phase of the game for the given player.
        """
        stones_on_board = len(self.current_player_stones_on_board(player))
        if self.hands[player] > 0:
            return "placing"
        elif stones_on_board > 3:
            return "moving"
        else:
            return "flying"

    def forms_mill(self, move_point: str, player: str) -> bool:
        """
        Checks if placing a stone at a given position forms a mill.
        """
        for trip in self.mill_combinations:
            if move_point in trip:
                if all((pt == move_point or self.board[pt] == player) for pt in trip):
                    return True
        return False

    def all_mills_of_player(self, player: str) -> List[Tuple[str, str, str]]:
        """
        Returns a list of all mills formed.
    """
        result = []
        for trip in self.mill_combinations:
            if all(self.board[pt] == player for pt in trip):
                result.append(trip)
        return result

    def stone_is_in_mill(self, pt: str, player: str) -> bool:
        """
        Checks if the given stone is part of an active mill for the specified player.
        """
        for trip in self.mill_combinations:
            if pt in trip:
                if all(self.board[x] == player for x in trip):
                    return True
        return False

    def generate_moves(self, player: str) -> List[Tuple[str, str, str]]:
        """
        Generates all legal moves for the given player based on the game phase.
        """
        moves = []
        opponent = self.opponent(player)
        phase = self.get_phase(player)

        # 1) If in "placing" phase
        if phase == "placing":
            if self.hands[player] > 0:
                for dest in self.valid_points():
                    if self.board[dest] is None:
                        if self.forms_mill(dest, player):
                            remove_candidates = self.get_remove_candidates(opponent)
                            for r in remove_candidates:
                                moves.append((f"h{1 if player=='blue' else 2}", dest, r))
                        else:
                            moves.append((f"h{1 if player=='blue' else 2}", dest, "r0"))
        else:
            # 'moving' or 'flying'
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
        """
        Gets a list of opponent's stones that can be removed.
        """
        opp_stones = self.current_player_stones_on_board(opponent)
        not_in_mill = [pt for pt in opp_stones if not self.stone_is_in_mill(pt, opponent)]
        return not_in_mill if len(not_in_mill) > 0 else opp_stones

    def forms_mill_after_move(self, src: str, dest: str, player: str) -> bool:
        """
        Check if moving a stone from `src` to `dest` forms a mill.
        """
        orig_src_occupant = self.board[src]
        self.board[src] = None  # remove from src
        mill_formed = self.forms_mill(dest, player)
        self.board[src] = orig_src_occupant  # revert
        return mill_formed

    def make_move(self, move: Tuple[str, str, str]) -> int:
        """
        Executes the given move and updates the game state.
        """
        old_no_mill = self.no_mill_moves
        src, dest, remove = move

        # Place or move the stone
        if src.startswith("h"):
            self.hands[self.turn] -= 1
            self.board[dest] = self.turn
        else:
            self.board[dest] = self.turn
            self.board[src] = None

        # Remove an opponent stone if needed
        if remove != "r0":
            self.board[remove] = None
            self.no_mill_moves = 0
        else:
            self.no_mill_moves += 1

        self.move_count += 1
        self.turn = self.opponent(self.turn)
        return old_no_mill

    def undo_move(self, move: Tuple[str, str, str], prev_turn: str, old_no_mill: int):
        """
        Undoes the given move to restore the previous game state.
        """
        src, dest, remove = move  #Extract move details.
        self.turn = prev_turn  #Revert turn to previous player.

        #If the stone was placed from the player's hand, return it to the hand.
        if src.startswith("h"):
            self.hands[prev_turn] += 1  #Restore the player's hand count.
            self.board[dest] = None  #Remove the stone from the board.
        else:
            self.board[src] = prev_turn  #Return the stone to its original position.
            self.board[dest] = None  #Clear the destination position.

        #Restore the opponent's stone if a removal occurred.
        if remove != "r0":
            opp = self.opponent(prev_turn)  #Get the opponent's identifier.
            self.board[remove] = opp  #Restore the opponent's stone to the board.

        #Restore the no-mill move counter.
        self.no_mill_moves = old_no_mill  
        self.move_count -= 1  #Decrease move count to reflect the undo move.

    def is_terminal_state(self) -> bool:
        """
        Checks if the game has reached a terminal state.
        For internal use if needed, but the referee also checks these conditions.
        """
        blue_total = len(self.current_player_stones_on_board("blue")) + self.hands["blue"]
        orange_total = len(self.current_player_stones_on_board("orange")) + self.hands["orange"]
        possible_moves = self.generate_moves(self.turn)

        # Basic checks (though the referee does the official checks)
        if blue_total <= 2 or orange_total <= 2:
            return True
        if not possible_moves:
            return True
        if self.no_mill_moves >= stalemate_threshold:
            return True
        return False

    def get_winner(self) -> Optional[str]:
        """
        Determines the winner of the game if the game has ended.
        Referee decides this in practice. Just here for potential local checks.
        """
        if not self.is_terminal_state():
            return None

        # Stalemate?
        if self.no_mill_moves >= stalemate_threshold:
            return 'draw'

        blue_total = len(self.current_player_stones_on_board("blue")) + self.hands["blue"]
        orange_total = len(self.current_player_stones_on_board("orange")) + self.hands["orange"]
        if blue_total <= 2 and orange_total > 2:
            return "orange"
        if orange_total <= 2 and blue_total > 2:
            return "blue"
        if not self.generate_moves(self.turn):
            return self.opponent(self.turn)
        return 'draw'

    # -------------------- EVAL & MINIMAX -------------------- #
    def evaluate(self, player: str) -> int:
        """
        Evaluates the board state and returns a heuristic score for the given player.
        """
        p_stones = len(self.current_player_stones_on_board(player))
        o_stones = len(self.current_player_stones_on_board(self.opponent(player)))
        score = (p_stones - o_stones) * 10

        p_mills = len(self.all_mills_of_player(player))
        o_mills = len(self.all_mills_of_player(self.opponent(player)))
        score += (p_mills - o_mills) * 2

        return score

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool, base_player: str
               ) -> Tuple[float, Optional[Tuple[str, str, str]]]:
        """
        Implements the minimax algorithm with alpha-beta pruning.

        Args:
        depth (int): The remaining depth to search in the game tree.
        alpha (float): The best value that the maximizing player can guarantee.
        beta (float): The best value that the minimizing player can guarantee.
        maximizing_player (bool): True if the current move is for the maximizing player, False for minimizing.
        base_player (str): The original player for whom the evaluation is performed.

        Returns:
        Tuple[float, Optional[Tuple[str, str, str]]]: 
        - The evaluated score of the board state.
        - The best move found in the format (source, destination, optional_removal), or None if no moves exist.
        """

        #Base case: If depth is 0 or the game is in a terminal state, evaluate the board.
        if depth == 0 or self.is_terminal_state():
            return self.evaluate(base_player), None

        current_player = self.turn  #Identify the player whose turn it is.
        moves = self.generate_moves(current_player)  #Get all possible moves.

        #If there are no valid moves, return the evaluation of the board.
        if not moves:
            return self.evaluate(base_player), None

        best_move = None #Store the best move found

        if maximizing_player:
            value = float('-inf') #Maximizing player's goal is to maximize score.
            for move in moves:
                old_no_mill = self.make_move(move) #Apply the move and store if a mill was formed.
                score, _ = self.minimax(depth - 1, alpha, beta, False, base_player) #Simulate opponent's turn with reduced depth.
                self.undo_move(move, current_player, old_no_mill) #Undo the move to restore state.

                #Update the best score and move if a higher score is found.
                if score > value:
                    value = score
                    best_move = move

                #Update alpha and perform pruning if needed.
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move #Return the best score and move for the maximizing player
        
        else: #Minimizing player's turn
            value = float('inf') #Minimizing player's goal is to minimize score.
            for move in moves:
                old_no_mill = self.make_move(move) #Apply the move and store if a mill was formed.
                score, _ = self.minimax(depth - 1, alpha, beta, True, base_player) #Simulate opponent's turn with reduced depth.
                self.undo_move(move, current_player, old_no_mill) #Undo the move to restore state.

                #Update the best score and move if a lower score is found.
                if score < value:
                    value = score
                    best_move = move
                #Update beta and perform pruning if needed.
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move #Return the best score and move for the minimizing player.

    def choose_best_move(self, max_depth: int = 6) -> Optional[Tuple[str, str, str]]:
        """
        Chooses the best move using minimax within the time limit.
        The maximum depth to search in the minimax algorithm is set to 6.
        """
        start = time.time()  #Record the start time.
        best = None  #Store the best move found.

        #Start at depth 1 and increase up to max_depth
        for depth in range(1, max_depth + 1):
            #Check if remaining time is too low to continue searching.
            if (time.time() - start) > time_limit - 0.05:
                break
            
            #Run minimax at the current depth.
            val, move = self.minimax(depth, float('-inf'), float('inf'), True, self.turn)
            
            #If there is still enough time left and a move was found, update the best move.
            if (time.time() - start) <= time_limit - 0.05 and move is not None:
                best = move
            else:
                break
        return best #Return the best move found within the time constraint.

    def do_our_move(self):
        """
        Computes our best move, prints it, and updates our internal board.
        If no moves are found, prints a pass-like move.
        """
        move = self.choose_best_move()
        if move is None:
            # No moves => must output something to the referee.
            # Using "X0" as an impossible board location for passing.
            print(f"h{1 if self.my_color=='blue' else 2} X0 r0", flush=True)
            return

        src, dest, remove = move
        # Print the move in the format "src dest remove"
        print(f"{src} {dest} {remove}", flush=True)
        self.make_move(move)

    # -------------------- REFEREE INTEGRATION -------------------- #
    def run_with_referee(self):
        """
        This Handles the communication protocol with the referee:
          1) Reads our assigned color ("blue" or "orange").
          2) Waits for moves from the opponent, applies them, then does our move.
          3) Stops if we receive "END:" or "Draw!".
        """
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line:
                    # No more input
                    break

                # 1) If line is "blue" or "orange", we store our color
                if line == "blue":
                    self.my_color = "blue"
                    self.opp_color = "orange"
                    self.turn = "blue"
                    # We are blue => we move first
                    self.do_our_move()

                elif line == "orange":
                    self.my_color = "orange"
                    self.opp_color = "blue"
                    # Blue goes first, so we do NOT move yet.
                    # We'll wait for the first "blue" move line
                    # But set turn = "blue" to apply the move properly
                    self.turn = "blue"

                # 2) If line starts with "END:" or is exactly "Draw!", the game is over
                elif line.startswith("END:") or line.startswith("Draw!"):
                    # The referee signals the game is finished
                    break

                else:
                    # 3) Otherwise, the line is an opponent's move in the format: "src dest remove"
                    tokens = line.split()
                    if len(tokens) == 3:
                        src, dest, remove = tokens
                        # Set self.turn = opp_color so that make_move is done from opponent's perspective
                        self.turn = self.opp_color
                        self.make_move((src, dest, remove))

                        # Now it's our turn
                        self.do_our_move()
                    else:
                        # Unknown line format, but typically should not happen under correct referee
                        pass

            except EOFError:
                break


    # -------------------- LOCAL TESTING GAME LOOP -------------------- #
    def play(self):
        """
        Runs a local testing game loop where the AI can play against itself or a human.
        """
        while True:
            print(f"\n=== Turn: {self.turn} ===")
            print("Board:", self.board)

            if self.is_terminal_state():
                winner = self.get_winner()
                if winner == 'draw':
                    print("Game ends in a draw.")
                else:
                    print(f"Game Over. Winner is {winner}.")
                break

            # Print possible moves for whoever is about to move
            all_moves = self.generate_moves(self.turn)
            print(f"Possible moves for {self.turn}: {all_moves}")

            move = self.choose_best_move()
            if move is None:
                # means search found no moves or timed out
                print(f"{self.turn} cannot move or times out. {self.opponent(self.turn)} wins!")
                break

            print(f"{self.turn} plays: {move}")
            self.make_move(move)

            if self.move_count > 200:
                print("Forcing stop after 200 moves.")
                break

# --------------- End of class definition ---------------

def main():
    game = LaskerMorris()
    # game.run_with_referee()
    game.play()

if __name__ == "__main__":
    main()
