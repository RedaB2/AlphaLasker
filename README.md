# Player222 🎮

This project is a Tic-tac-toe player named **Player222** that uses the minimax algorithm with alpha-beta pruning. It is designed to play optimally and communicate with the referee system. This is Part I of the project. 

## 1. Instructions on Compiling and Running the Program 🔧

### Requirements
- **Python Version:** Python 3.10 or higher  
- **Dependencies:** No external packages are required beyond the standard Python libraries (sys, math).

### Running with the Referee
To run the program with the cs4341-referee, use the following command in your terminal:

```bash
cs4341-referee tictactoe -p "python player222.py" --visual
```

This command will allow to start the game with the referee, turn on visualization, and run the program player222.py

### Running with offline tests
In order to run the program and run local tests with preset boards, uncomment the following line at the end of the code: 

```bash 
    run_offline_tests()
```


## 2. evalTerminalState(boardState, myMark)

The AI uses evalTerminalState(boardState, myMark) function as its utility function. It operates as follows:

- Winning Terminal States:
    If the board shows that your player (e.g., 'X' for blue) has won, the function returns +100.
- Losing Terminal States:
    If the board shows that the opponent has won, the function returns -100.
-  Draw or Non-Terminal States:
    If there is no winner (including draws), it returns 0.

## 3. Results

Testing: 

1. Offline Testing 
- Test 1: Empty Board
The program was tested on an empty board with 'X' to move first. The AI chose a valid move.
- Test 2: Partially-Filled Board
A board scenario was constructed where moves had already been made. The AI selected a move that appropriately blocked or created a threat.
- Test 3: Near-Terminal Board
A scenario was tested where 'X' could win by playing a specific move. The AI correctly identified and selected the winning move.

2. Self-Play via the Referee:
The AI was run against itself using the referee system.
As expected for optimal play in Tic-tac-toe, every game ends in a draw.

## 3. Strengths 💪

- Optimal Play:
The use of minimax with alpha-beta pruning ensures optimal play. Therefore, we get a draw. 
- Robust Move Generation:
The program validates moves, updates its internal board state accurately, and communicates only valid moves to the referee.
- Efficient Search:
The search algorithm is efficient for a simple game like Tic-tac-toe and always returns a move well within the allotted time limit.
- Clear Code Structure:
The code is modular and well-commented, making it easy to understand, maintain, and extend.

## 4. Weaknesses ⚠️

- Simple Evaluation Heuristic:
The utility function is basic (+100, -100, or 0). This is enough for Tic Tac Toe, however, might not be enough for more complex.
- Scalability:
The full minimax search works well for Tic-tac-toe, but scaling this approach to more complex games would require further optimizations (such as move ordering or iterative deepening).