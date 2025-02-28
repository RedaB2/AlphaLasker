import sys
import time
import re
import random
import traceback  
from typing import List, Tuple, Dict, Optional
from collections import deque
from google import genai


llm_client = genai.Client(api_key="AIzaSyCMRApVKh7qf_jiRv3onxoaKT-8-DSoWGY")


time_limit = 60           
stalemate_threshold = 20  

request_timestamps = deque()

def throttled_generate_content(model: str, contents: str, timeout: int = 60):
    """
    Waits if necessary so that no more than 15 requests are made in a rolling 60-second window.
    Then calls the Gemini API.
    """
    global request_timestamps
    current_time = time.time()
    # Remove timestamps older than 60 seconds.
    while request_timestamps and current_time - request_timestamps[0] > 60:
        request_timestamps.popleft()
    if len(request_timestamps) >= 15:
        wait_time = 60 - (current_time - request_timestamps[0])
        print(f"DEBUG: Rate limit reached. Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
    request_timestamps.append(time.time())
    print(f"DEBUG: Sending request to Gemini API...")
    try:
        response = llm_client.models.generate_content(model=model, contents=contents)
        print(f"DEBUG: Received response from Gemini API")
        return response
    except Exception as e:
        print(f"DEBUG: Error in API call: {str(e)}")
        raise

class LaskerMorris:
    """
    Implements the game logic and LLM-based AI for Lasker Morris.
    All move selection is solely performed by prompting the Gemini LLM.
    If the LLM fails to produce a valid move after two attempts, a fallback move is chosen.
    """
    def __init__(self):
        self.board: Dict[str, Optional[str]] = self.initialize_board()
        self.hands = {'blue': 10, 'orange': 10}
        self.turn = 'blue'
        self.no_mill_moves = 0
        self.move_count = 0  # tracks total moves made
        self.my_color: Optional[str] = None
        self.opp_color: Optional[str] = None
        self.last_move: Optional[Tuple[str, str, str]] = None  # last move played
        
        self.adjacency_list = self.build_adjacency_list()
        self.mill_combinations = self.build_mill_combinations()

    def initialize_board(self) -> Dict[str, Optional[str]]:
        board = {}
        for pt in self.valid_points():
            board[pt] = None
        return board

    def valid_points(self) -> List[str]:
        return [
            "a7", "d7", "g7", "b6", "d6", "f6", 
            "c5", "d5", "e5", "a4", "b4", "c4", 
            "e4", "f4", "g4", "c3", "d3", "e3", 
            "b2", "d2", "f2", "a1", "d1", "g1"
        ]

    def build_adjacency_list(self) -> Dict[str, List[str]]:
        return {
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
            "d2": ["b2", "f2", "d3", "d1"],
            "f2": ["d2", "f4"],
            "a1": ["a4", "d1"],
            "d1": ["a1", "g1", "d2"],
            "g1": ["d1", "g4"]
        }

    def build_mill_combinations(self) -> List[Tuple[str, str, str]]:
        mills = [
            ("a7", "d7", "g7"),
            ("b6", "d6", "f6"),
            ("c5", "d5", "e5"),
            ("a4", "b4", "c4"),
            ("e4", "f4", "g4"),
            ("c3", "d3", "e3"),
            ("b2", "d2", "f2"),
            ("a1", "d1", "g1"),
            ("a7", "a4", "a1"),
            ("b6", "b4", "b2"),
            ("c5", "c4", "c3"),
            ("d7", "d6", "d5"),
            ("d3", "d2", "d1"),
            ("e5", "e4", "e3"),
            ("f6", "f4", "f2"),
            ("g7", "g4", "g1")
        ]
        return [tuple(sorted(trip)) for trip in mills]

    def current_player_stones_on_board(self, player: str) -> List[str]:
        return [pt for pt, occupant in self.board.items() if occupant == player]

    def opponent(self, player: str) -> str:
        return "blue" if player == "orange" else "orange"

    def get_phase(self, player: str) -> str:
        stones = len(self.current_player_stones_on_board(player))
        if self.hands[player] > 0:
            return "placing"
        elif stones > 3:
            return "moving"
        else:
            return "flying"

    def forms_mill(self, pt: str, player: str) -> bool:
        for trip in self.mill_combinations:
            if pt in trip and all((p == pt or self.board[p] == player) for p in trip):
                return True
        return False

    def all_mills_of_player(self, player: str) -> List[Tuple[str, str, str]]:
        return [trip for trip in self.mill_combinations if all(self.board[p] == player for p in trip)]

    def stone_is_in_mill(self, pt: str, player: str) -> bool:
        for trip in self.mill_combinations:
            if pt in trip and all(self.board[p] == player for p in trip):
                return True
        return False

    def generate_moves(self, player: str) -> List[Tuple[str, str, str]]:
        print(f"DEBUG: Generating moves for {player}. Phase: {self.get_phase(player)}")
        moves = []
        opp = self.opponent(player)
        phase = self.get_phase(player)

        if phase == "placing":
            if self.hands[player] > 0:
                for dest in self.valid_points():
                    if self.board[dest] is None:
                        if self.forms_mill(dest, player):
                            for r in self.get_remove_candidates(opp):
                                moves.append((f"h{1 if player=='blue' else 2}", dest, r))
                        else:
                            moves.append((f"h{1 if player=='blue' else 2}", dest, "r0"))
        else:
            stones = self.current_player_stones_on_board(player)
            print(f"DEBUG: Player {player} has {len(stones)} stones on board")
            for src in stones:
                if phase == "moving":
                    dests = [p for p in self.adjacency_list[src] if self.board[p] is None]
                else:
                    dests = [p for p in self.board if self.board[p] is None]
                for dest in dests:
                    if self.forms_mill_after_move(src, dest, player):
                        for r in self.get_remove_candidates(opp):
                            moves.append((src, dest, r))
                    else:
                        moves.append((src, dest, "r0"))
        print(f"DEBUG: Generated {len(moves)} possible moves")
        return moves

    def get_remove_candidates(self, opponent: str) -> List[str]:
        opp_stones = self.current_player_stones_on_board(opponent)
        non_mill = [pt for pt in opp_stones if not self.stone_is_in_mill(pt, opponent)]
        return non_mill if non_mill else opp_stones

    def forms_mill_after_move(self, src: str, dest: str, player: str) -> bool:
        orig = self.board[src]
        self.board[src] = None
        mill = self.forms_mill(dest, player)
        self.board[src] = orig
        return mill

    def make_move(self, move: Tuple[str, str, str]) -> int:
        old_no_mill = self.no_mill_moves
        src, dest, remove = move
        print(f"DEBUG: Making move {move} for player {self.turn}")

        if src.startswith("h"):
            self.hands[self.turn] -= 1
            self.board[dest] = self.turn
        else:
            self.board[dest] = self.turn
            self.board[src] = None

        if remove != "r0":
            self.board[remove] = None
            self.no_mill_moves = 0
        else:
            self.no_mill_moves += 1

        self.move_count += 1
        self.turn = self.opponent(self.turn)
        print(f"DEBUG: Move complete. Next turn: {self.turn}")
        return old_no_mill

    def undo_move(self, move: Tuple[str, str, str], prev_turn: str, old_no_mill: int):
        src, dest, remove = move
        self.turn = prev_turn
        if src.startswith("h"):
            self.hands[prev_turn] += 1
            self.board[dest] = None
        else:
            self.board[src] = prev_turn
            self.board[dest] = None
        if remove != "r0":
            opp = self.opponent(prev_turn)
            self.board[remove] = opp
        self.no_mill_moves = old_no_mill
        self.move_count -= 1

    def is_terminal_state(self) -> bool:
        blue_total = len(self.current_player_stones_on_board("blue")) + self.hands["blue"]
        orange_total = len(self.current_player_stones_on_board("orange")) + self.hands["orange"]
        print(f"DEBUG: Checking terminal state. Blue total: {blue_total}, Orange total: {orange_total}, No mill moves: {self.no_mill_moves}")
        
        if blue_total <= 2 or orange_total <= 2:
            print(f"DEBUG: Terminal state - player has fewer than 3 stones")
            return True
            
        if not self.generate_moves(self.turn):
            print(f"DEBUG: Terminal state - no valid moves for current player")
            return True
            
        if self.no_mill_moves >= stalemate_threshold:
            print(f"DEBUG: Terminal state - stalemate threshold reached")
            return True
            
        return False

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal_state():
            return None
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

    def validate_move(self, move: Tuple[str, str, str]) -> bool:
        valid_moves = self.generate_moves(self.turn)
        is_valid = move in valid_moves
        print(f"DEBUG: Validating move {move} for {self.turn}: {'Valid' if is_valid else 'Invalid'}")
        return is_valid

    def get_llm_move(self) -> Optional[Tuple[str, str, str]]:
        """
        Constructs a prompt that describes the game state and rules,
        sends it to the Gemini LLM (using our rate-limited function),
        and extracts the move.
        If the move does not pass validation on the first attempt,
        a second (error) prompt is sent.
        If both attempts fail, a fallback valid move is chosen.
        """
        print(f"DEBUG: Getting LLM move for {self.turn}")
        board_state = ", ".join(f"{pt}:{self.board[pt] if self.board[pt] else '.'}" for pt in self.valid_points())
        hands_state = f"Blue hand: {self.hands['blue']}, Orange hand: {self.hands['orange']}"
        last_move_str = f"{self.last_move}" if self.last_move else "None"
        
        prompt = (
            f"You are an AI that plays Lasker Morris. The rules are:\n"
            f"- Each player starts with 10 stones (blue uses h1, orange uses h2).\n"
            f"- Valid board points: {', '.join(self.valid_points())}.\n"
            f"- Moves must be in the format (src dest removal), for example (h1 a4 r0).\n"
            f"Current board state: {board_state}.\n"
            f"Hands: {hands_state}.\n"
            f"Last move played: {last_move_str}.\n"
            f"It is player {self.turn}'s turn.\n"
            f"Output a single line containing the next valid move in the format (src dest removal)."
        )
        
        move_tuple = None
        try:
            print(f"DEBUG: Sending first prompt to LLM")
            response = throttled_generate_content(model="gemini-2.0-flash", contents=prompt)
            response_text = response.text
            print(f"DEBUG: LLM response: {response_text[:100]}...")
            match = re.search(r'\(([^\)]+)\)', response_text)
            if match:
                tokens = match.group(1).strip().split()
                print(f"DEBUG: Extracted tokens: {tokens}")
                if len(tokens) == 3:
                    move_tuple = tuple(tokens)
                    if self.validate_move(move_tuple):
                        print(f"DEBUG: Valid move found: {move_tuple}")
                        return move_tuple
                    else:
                        print(f"DEBUG: Move {move_tuple} is not valid")
                else:
                    print(f"DEBUG: Incorrect number of tokens: {len(tokens)}")
            else:
                print(f"DEBUG: No move pattern found in response")
        except Exception as e:
            print(f"DEBUG: Exception in first LLM attempt: {str(e)}")
            traceback.print_exc()

        error_prompt = (
            f"Your previous move output was invalid or could not be parsed. "
            f"Please try again.\n"
            f"Board state: {board_state}.\n"
            f"Hands: {hands_state}.\n"
            f"Last move played: {last_move_str}.\n"
            f"It is player {self.turn}'s turn.\n"
            f"Output a valid move in the format (src dest removal)."
        )
        try:
            print(f"DEBUG: Sending second prompt to LLM")
            response = throttled_generate_content(model="gemini-2.0-flash", contents=error_prompt)
            response_text = response.text
            print(f"DEBUG: Second LLM response: {response_text[:100]}...")
            match = re.search(r'\(([^\)]+)\)', response_text)
            if match:
                tokens = match.group(1).strip().split()
                print(f"DEBUG: Extracted tokens (second attempt): {tokens}")
                if len(tokens) == 3:
                    move_tuple = tuple(tokens)
                    if self.validate_move(move_tuple):
                        print(f"DEBUG: Valid move found (second attempt): {move_tuple}")
                        return move_tuple
                    else:
                        print(f"DEBUG: Move {move_tuple} is not valid (second attempt)")
                else:
                    print(f"DEBUG: Incorrect number of tokens (second attempt): {len(tokens)}")
            else:
                print(f"DEBUG: No move pattern found in second response")
        except Exception as e:
            print(f"DEBUG: Exception in second LLM attempt: {str(e)}")
            traceback.print_exc()

        ## FALLBACK
        valid_moves = self.generate_moves(self.turn)
        if valid_moves:
            fallback_move = random.choice(valid_moves)
            print(f"DEBUG: Fallback logic used. Selected move: {fallback_move}")
            print("Fallback logic used: LLM move invalid. Using fallback move:", fallback_move, file=sys.stderr)
            return fallback_move
        else:
            print("DEBUG: No valid moves available for fallback")
            return None

    def do_our_move(self):
        """
        Uses only the LLM (via our rate-limited calls) to generate our move.
        If the LLM fails twice to produce a valid move, the fallback move is used.
        """
        print(f"DEBUG: do_our_move called for {self.turn}")
        move = self.get_llm_move()
        if move is None:
            print("DEBUG: LLM failed to generate a move and no fallback move is available.")
            sys.exit("LLM failed to generate a move and no fallback move is available.")
        src, dest, remove = move
        print(f"{src} {dest} {remove}", flush=True)
        self.make_move(move)

    # ---------------- REFEREE COMMUNICATION ---------------- #
    def run_with_referee(self):
        """
        Communicates with the referee: reads color assignment, opponent moves,
        and makes moves when it is our turn.
        """
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line:
                    break

                if line == "blue":
                    self.my_color = "blue"
                    self.opp_color = "orange"
                    self.turn = "blue"
                    self.last_move = None
                    self.do_our_move()

                elif line == "orange":
                    self.my_color = "orange"
                    self.opp_color = "blue"
                    self.turn = "blue"

                elif line.startswith("END:") or line.startswith("Draw!"):
                    break

                else:
                    tokens = line.split()
                    if len(tokens) == 3:
                        src, dest, remove = tokens
                        self.last_move = (src, dest, remove)
                        self.turn = self.opp_color
                        self.make_move((src, dest, remove))
                        self.do_our_move()
                    else:
                        pass

            except EOFError:
                break

    def play(self):
        """
        Local testing loop that displays the board state and uses the LLM to move.
        """
        try:
            print("DEBUG: Starting local testing game loop")
            while True:
                print(f"\n=== Turn: {self.turn} ===")
                print(f"DEBUG: Game state - Move count: {self.move_count}, No mill moves: {self.no_mill_moves}")
                
                # Print a clearer game state summary
                blue_stones = len(self.current_player_stones_on_board("blue"))
                orange_stones = len(self.current_player_stones_on_board("orange"))
                print(f"GAME STATE: Blue has {blue_stones} stones on board + {self.hands['blue']} in hand = {blue_stones + self.hands['blue']} total")
                print(f"GAME STATE: Orange has {orange_stones} stones on board + {self.hands['orange']} in hand = {orange_stones + self.hands['orange']} total")
                print(f"GAME STATE: Current phase for {self.turn}: {self.get_phase(self.turn)}")
                
                print("Board:", self.board)
                
                # Print a more readable board representation
                print("\nBoard visualization:")
                self.print_board()

                if self.is_terminal_state():
                    winner = self.get_winner()
                    print("\n========== GAME OVER ==========")
                    if winner == "draw":
                        print("Game ends in a draw.")
                    else:
                        print(f"Winner is {winner}!")
                        
                    # Print game summary
                    print("\nFinal Game Summary:")
                    print(f"Total moves played: {self.move_count}")
                    print(f"Blue stones remaining: {blue_stones + self.hands['blue']}")
                    print(f"Orange stones remaining: {orange_stones + self.hands['orange']}")
                    
                    if winner == "blue":
                        print("Blue won by reducing Orange to fewer than 3 stones or blocking all Orange moves.")
                    elif winner == "orange":
                        print("Orange won by reducing Blue to fewer than 3 stones or blocking all Blue moves.")
                    else:
                        print(f"Game ended in a draw after {self.no_mill_moves} moves without mills forming.")
                    
                    print("================================")
                    break

                valid_moves = self.generate_moves(self.turn)
                print(f"DEBUG: Number of possible moves: {len(valid_moves)}")
                if len(valid_moves) > 10:
                    print(f"Possible moves for {self.turn}: {valid_moves[:10]} ... (and {len(valid_moves)-10} more)")
                else:
                    print(f"Possible moves for {self.turn}: {valid_moves}")
                
                print(f"DEBUG: Getting move for {self.turn}")
                move = self.get_llm_move()
                if move is None:
                    print("DEBUG: LLM failed to generate a move during testing.")
                    sys.exit("LLM failed to generate a move during testing.")
                print(f"{self.turn} plays: {move}")
                self.make_move(move)
                if self.move_count > 200:
                    print("Forcing stop after 200 moves.")
                    break
        except Exception as e:
            print(f"DEBUG: Exception in play loop: {str(e)}")
            traceback.print_exc()
            
    def print_board(self):
        """Print a more readable representation of the board"""
        board_template = """
        7   {}-----{}-----{}
            |       |       |
        6   |   {}---{}---{}   |
            |   |       |   |
        5   |   |   {}---{}---{}   |   |
            |   |   |       |   |   |
        4   {}---{}---{}       {}---{}---{}
            |   |   |       |   |   |
        3   |   |   {}---{}---{}   |   |
            |   |       |   |
        2   |   {}---{}---{}   |
            |       |       |
        1   {}-----{}-----{}
            a   b   c   d   e   f   g
        """
        
    
        board_values = []
        for point in ["a7", "d7", "g7", "b6", "d6", "f6", "c5", "d5", "e5", 
                      "a4", "b4", "c4", "e4", "f4", "g4", "c3", "d3", "e3", 
                      "b2", "d2", "f2", "a1", "d1", "g1"]:
            value = self.board[point]
            if value is None:
                board_values.append("·")
            elif value == "blue":
                board_values.append("B")
            else:
                board_values.append("O")
                
        print(board_template.format(*board_values))

def main():
    try:
        print("DEBUG: Starting AlphaLasker")
        game = LaskerMorris()
        # To run with the referee, uncomment the following line:
        game.run_with_referee()
        # For local testing, use:
        # game.play()
    except Exception as e:
        print(f"DEBUG: Exception in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()