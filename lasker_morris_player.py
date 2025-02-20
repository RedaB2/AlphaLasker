import sys
from typing import Tuple, Optional
from AlphaLasker import LaskerMorris  

class LaskerMorrisPlayer:
    def __init__(self):
        self.game = LaskerMorris()
        self.player_id = None  

    def parse_move(self, move_str: str) -> Tuple[str, str, str]:
        """Parse a move string from the referee into a tuple (source, target, remove)."""
        parts = move_str.strip().split()
        if len(parts) == 3:
            return tuple(parts)
        raise ValueError(f"Invalid move format: {move_str}")

    def format_move(self, move: Tuple[str, str, str]) -> str:
        """Format a move tuple into a string for the referee."""
        return " ".join(move)

    def run(self):
        """Main loop to interact with the referee."""
        try:
            
            self.player_id = input().strip()
            print(f"Player ID: {self.player_id}", file=sys.stderr)

            while True:
                
                opponent_move_str = input().strip()
                if opponent_move_str:
                    
                    opponent_move = self.parse_move(opponent_move_str)
                    self.game.make_move(opponent_move)

               
                our_move = self.game.best_move()  
                if our_move is None:
                    print("No valid moves left.", file=sys.stderr)
                    break

                
                move_str = self.format_move(our_move)
                print(move_str, flush=True)

                
                self.game.make_move(our_move)

        except EOFError:
            
            print("Game over.", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    player = LaskerMorrisPlayer()
    player.run()