# Checkers-Solver-
A UOFT CSC384 project, utilizing a game tree search algorithm with a alpha beta pruning and depth limiting functionality.

# Checker Solver Using Alpha-Beta Pruning

This project is an **AI-based checker solver** that uses the **Alpha-Beta Pruning** algorithm to efficiently evaluate and make optimal moves in a game of checkers. It is designed to minimize computational overhead while exploring game states and provide strategic decision-making for competitive gameplay.

---

## Features

- Implements **Alpha-Beta Pruning** to optimize the minimax algorithm for checkers.
- Efficient evaluation of game states with a custom **heuristic function**.
- Supports configurable depth for tree exploration, balancing speed and accuracy.
- Detects valid moves, captures, and game termination conditions.
- Handles both **human vs. AI** and **AI vs. AI** gameplay modes.

---

## How It Works

The solver uses the **Alpha-Beta Pruning algorithm**, which enhances the classic minimax approach by eliminating unnecessary branches in the game tree. This significantly reduces the computational time needed to determine the best possible move for the current player.

1. **Game Tree Representation**:
   - Each node represents a game state.
   - The edges represent possible moves.

2. **Evaluation Function**:
   - A heuristic evaluates the desirability of a game state, factoring in:
     - Piece count (regular pieces vs. kings).
     - Positional advantage.
     - Mobility and potential captures.

3. **Alpha-Beta Pruning**:
   - Alpha tracks the **best score** achievable by the maximizer (AI).
   - Beta tracks the **best score** achievable by the minimizer (opponent).
   - Subtrees that cannot influence the final decision are skipped.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/checker-solver.git
   cd checker-solver