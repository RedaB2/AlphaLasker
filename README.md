# AlphaLasker: Lasker Morris AI with Gemini API

## Team Members and Contributions

Carlos Jones ==> LLM implementation, testing, documentation, comments

Reda Boutayeb == LLM implementation, referee communication, testing

## Table of Contents
- [Introduction](#introduction)
- [Compilation and Execution](#compilation-and-execution)
- [System Description](#system-description)
- [Prompt Engineering](#prompt-engineering)
- [Testing & Results](#testing--results)
- [Conclusion](#conclusion)

---

## Introduction
AlphaLasker is an AI player for the Lasker Morris game that leverages Google's Gemini 2.0 Flash API to evaluate board states and make optimal moves. It competes against a minimax-based AI from a previous project and human players to assess its effectiveness as a game-playing agent.

## Compilation and Execution
### Requirements
- Python 3.x
- `openai` or `google-generativeai` Python package (for API calls)
- API key for Gemini 2.0 Flash

### Installation

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Set up API key securely:
   ```sh
   export GEMINI_API_KEY="your_api_key_here"
   ```
   Or store it in a `.env` file and load it in your script.
3. Run the program:
   ```sh
   python AlphaLasker.py
   ```

## System Description
AlphaLasker interacts with the Gemini API to determine moves. The process follows these steps:
1. **Board State Encoding**: The current game state is converted into a structured prompt.
2. **API Interaction**:
   - The AI sends the board state and asks Gemini for the best move.
   - The response is parsed, extracting the suggested move.
3. **Move Validation**:
   - If the suggested move is invalid, the AI requests an alternative move.
   - If repeated invalid moves occur, the AI falls back to a heuristic-based decision.
4. **Game Execution**: The valid move is executed, and the game continues until completion.

## Prompt Engineering
### Prompts Used
The program employs structured prompts to guide the LLM in move selection. A sample prompt:
```
You are an expert Lasker Morris player. The current board state is:
[Board Representation]
What is the best move to make? Respond with the move in the format (start_position, end_position).
```

### Findings from Experimentation
- **Specificity matters**: Providing clear instructions, such as expected response format, significantly improved the model’s accuracy.
- **Contextual Memory**: The LLM sometimes forgot previous moves, requiring the inclusion of turn history in the prompt.
- **Handling Invalid Moves**: Adding explicit constraints in the prompt reduced invalid move occurrences but didn’t eliminate them completely.
- **Prompt Refinements**:
  - Initially, generic prompts resulted in illegal moves.
  - Including rule-based clarifications improved performance.
  - A dynamic prompt that adjusts based on previous mistakes showed the best results.

## Testing & Results
### AI vs Minimax AI (Project 1)
| Test | Winner |
|------|--------|
| 1    | AlphaLasker |
| 2    | Minimax AI |
| 3    | AlphaLasker |

AlphaLasker won ~60% of the matches, with better performance in mid-to-late game moves.

## Conclusion
- **Effectiveness**: The LLM can play at an intermediate level but is inconsistent in handling complex scenarios.
- **Limitations**:
  - Occasional invalid move suggestions.
  - Limited strategic foresight compared to minimax.
- **Potential Improvements**:
  - Fine-tuning the prompt.
  - Integrating a hybrid approach (LLM + rule-based checks).

### Final Verdict
While an LLM can be a competent game-playing agent, it lacks the precision and reliability of a dedicated algorithm like minimax for Lasker Morris. A hybrid approach combining both methodologies may yield the best results.

