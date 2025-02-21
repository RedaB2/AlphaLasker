# AlphaLasker 🎮

This project is a Lasker Morris player named **AlphaLasker** that uses the minimax algorithm with alpha-beta pruning. It is designed to play optimally and communicate with the referee system. This is Part I of the project. 


## Team Members & Contributions

Carlos Jones ==> Game logic and board management, debugging and testing, implementation of the evaluation function, minimax/alpha-beta, comments

Reda Boutayeb ==> Game logic, heuristic refinement, evaluation function, testing and local play, referee communication, comments

## 2. Description and architecture

AlphaLasker uses: 

- A state-based approach, different phases ==> placing, moving, flying. 
- A minimax search with alpha-beta pruning, which explores possible moves up to a given depth or until a time limit is reached. 
- A heuristic evaluation function, which estimates the desirability of any given board state based on stone counts and mills formed. 

The code is organized in: 

LaskerMorris class: Implements all the game mechanics, board, move, and search. 

main() : Simple entry poiunt for AI or referee. 



## 3. Instructions on Compiling and Running the Program 🔧

### Requirements
- **Python Version:** Python 3.10 or higher  
- **Dependencies:** No external packages are required beyond the standard Python libraries (sys, math, time, typing).

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

## 4. evaluate(self, player: str) - Utility Function/Evaluation

Within the code, the evaluate(self, player: str) method acts as the utility (heuristic) function for that player. It returns an integer score indicating how favorable the current board is to that player.

The key factors are: 

- Stone Count difference


- Number of Mills 

Thus, if a player has more stones and/or more active mills, it yields a higher score. 

## 5. Heuristics & Strats

- Phased Move Generation:
The code only generates moves relevant to the current phase for the active player (placing, moving, or flying).

- Alpha-Beta Pruning:
We prune branches in minimax if they cannot possibly influence the final decision, reducing the search space.

- Iterative Deepening:
The AI begins at depth 1 and increases the depth up to 6 or until the time limit (5 seconds) is about to be exceeded. The best move found so far is retained if the timer is close to the limit.

- Stalemate Threshold:
If there have been 20 moves in a row with no mills formed, the game is considered a draw.



## 7. Results & Testing


Local Testing
	•	AI vs AI (self-play) without a referee in game.play() mode:
	•	In repeated tests, the outcome was fairly consistent: orange would typically win. After a certain point, blue drops below 2 stones, causing orange to be the victor.
	•	Because the scenario plays out identically each time (no randomness in the search), the results are reproducible.

Referee Testing
	•	When played via the cs4341-referee, the AI runs smoothly and meets time constraints (under 5 seconds).
	•	Occasional timeouts can occur if the depth is set too high, but our iterative deepening approach mitigates this.


## 3. Strengths 💪

- Robust Move Generation: Correct handling of the placing, moving, and flying phases; accurate mill detection.
- Efficient Search: Alpha-beta pruning combined with iterative deepening keeps the search within time limits.
- Modular Code: Easy to read, debug, and maintain.
- Solid Heuristic: Stone count and mills cover the main tactical considerations in Lasker Morris.

## 4. Weaknesses ⚠️

- Heuristic Complexity: The evaluation is relatively simple and does not consider more sophisticated positional factors.
- No Advanced Learning: The AI does not store or reuse previously computed board states.
- Timeout at the end of game with referee. 
