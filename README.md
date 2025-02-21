# AlphaLasker 🎮

This project is a Lasker Morris player named **AlphaLasker** that uses the minimax algorithm with alpha-beta pruning. It is designed to play optimally and communicate with the referee system. This is Part I of the project. 

## 1. Instructions on Compiling and Running the Program 🔧

### Requirements
- **Python Version:** Python 3.10 or higher  
- **Dependencies:** No external packages are required beyond the standard Python libraries (sys, math).

### Running with the Referee
To run the program with the cs4341-referee, use the following command in your terminal:

```bash
cs4341-referee laskermorris -p1 "python AlphaLasker.py" -p2 "python AlphaLasker.py" --visual
```

This command will allow to start the game with the referee, turn on visualization, and run the program AlphaLasker.py

### Running with offline tests
In order to run the program and run local tests with preset boards, uncomment the following line at the end of the code: 

```bash 
    game.play()
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

- Robust Move Generation:
The program implements move generation, mills detection, and adjacency lists effectively.
- Efficient Search:
The search algorithm is efficient for a game like LaskerMorris and always returns a move within the allotted time limit.
- Clear Code Structure:
The code is modular and well-commented, making it easy to understand, maintain, and extend.
- Minimax Functionality:
The minimax function incorporates alpha-beta pruning, reducing the number of nodes that need to be evaluated.
- Heuristic Evaluation:
The evaluate function considers multiple factors such as stone count and mills, providing a reasonable heuristic for move selection.

## 4. Weaknesses ⚠️

- Heuristic Complexity:
The evaluation function is relatively simple, relying on a basic difference in stone count and mills.
- Referee Implementation:
In testing, the referee would occasionally time out after all moves were completed.
- Learning Mechanism:
There is no learning mechanism (e.g., transposition tables) to avoid re-evaluating previously seen board states.
