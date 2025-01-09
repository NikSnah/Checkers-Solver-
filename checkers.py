import argparse
import copy
import sys

# python3 checkers.py --inputfile Examples/checkers1.txt --outputfile Sols/checkers1_sol.txt
# python3 checkers.py --inputfile Examples/checkers0.txt --outputfile Sols/checkers0_sol.txt
# python3 checkers.py --inputfile Examples/checkers2.txt --outputfile Sols/checkers2_sol.txt
# python3 checkers.py --inputfile Examples/checkers3.txt --outputfile Sols/checkers3_sol.txt
# python3 checkers.py --inputfile Examples/checkers4.txt --outputfile Sols/checkers4_sol.txt

# Define constants for large win/loss values
WIN_SCORE = 1000000
LOSS_SCORE = -1000000

# this will be adaptive based on the number of pieces on the board
depth_limit = 7  # Set an initial depth limit
    

cache = {} # you can use this to implement state caching

class Piece: 

    def __init__(self, player, is_king, is_empty, row, col):
        self.player = player
        self.is_king = is_king
        self.is_empty = is_empty
        self.row = row
        self.col = col

    def __str__ (self):
        return self.player + " " + str(self.is_king) + " " + str(self.is_empty) + " " + str(self.row) + " " + str(self.col)

class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    def __init__(self, empty_pieces, red_pieces, black_pieces):

        self.string_board = ""
        self.empty_pieces = empty_pieces
        self.red_pieces = red_pieces
        self.black_pieces = black_pieces

        self.width = 8
        self.height = 8

        # number of normal peices on the board
        self.r_num = 0 
        self.b_num = 0

        # number of kings on the board
        self.r_king = 0
        self.b_king = 0

        self.update_player_count()
 
    def __str__(self):
        if self.string_board == "":
            for i in range(8):
                count = 0 
                for j in range(8):
                    if (i, j) in self.red_pieces:
                        self.string_board += self.red_pieces[(i, j)].player
                    elif (i, j) in self.black_pieces:
                        self.string_board += self.black_pieces[(i, j)].player
                    elif (i, j) in self.empty_pieces:
                        self.string_board += self.empty_pieces[(i, j)].player
                    else: 
                        return "Error creating string board"
                    
                    count += 1
                    if count == 8:
                        self.string_board += "\n"
        return self.string_board
        
    def update_player_count(self):
        self.r_num = 0
        self.b_num = 0
        self.r_king = 0
        self.b_king = 0
        for i in self.red_pieces.keys():
                if not self.red_pieces[i].is_king:
                    self.r_num += 1
                else:
                    self.r_king += 1
        for i in self.black_pieces.keys():
                if not self.black_pieces[i].is_king:
                    self.b_num += 1
                else:
                    self.b_king += 1
        return

def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']

def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'

def read_from_file(filename):

    f = open(filename)
    lines = f.readlines()
    empty_pieces = {}
    red_pieces = {}
    black_pieces = {}
    for row in range(len(lines)):
        for col in range(len(lines[row].rstrip())):
            # Check if its a king or normal piece/empty
            piece = None
            if lines[row][col] in ['R', 'B']:
                piece = Piece(lines[row][col], True, False, row, col)
            elif lines[row][col] in ['r', 'b']:
                piece = Piece(lines[row][col], False, False, row, col)
            else:
                empty_pieces[(row, col)] = Piece(lines[row][col], False, True, row, col)

            if lines[row][col] in ['r', 'R']:
                red_pieces[(row, col)] = piece
            elif lines[row][col] in ['b', 'B']:
                black_pieces[(row, col)] = piece

    f.close()

    return empty_pieces, red_pieces, black_pieces

def adaptive_depth_limit(state: State, move_number: int) -> int:
    """
    Adjust depth limit based on the game state.
    """
    total_pieces = state.r_num + state.b_num
    if total_pieces > 16:
        return 5  # Early game: smaller depth limit
    elif total_pieces > 8:
        return 7  # Mid game: moderate depth limit
    else:
        return 10  # End game: deeper search

def alpha_beta(state: State, depth: int, alpha: float, beta: float, maximizing_player: bool, player: str, depth_limit: int) -> tuple:
    """
    Alpha-Beta Pruning algorithm to search the game tree.
    Returns the best evaluation and the corresponding best move.
    """

    # Base case: terminal state or depth limit
    if depth == depth_limit or game_over(state):

        eval_value = evaluate(state, depth, player)
        #print(f"Eval: {eval_value}, Depth: {depth}, Player: {player}")
        return eval_value, None  # No move at terminal statestate

    best_move = None

    if maximizing_player:
        max_eval = -float('inf')
        # make move return a list of ready to go states 
        moves = get_valid_moves(state, player)
        for move in moves:
            eval, _ = alpha_beta(move, depth + 1, alpha, beta, False, get_next_turn(player), depth_limit)
            #print(f"Eval: {eval}, Max Eval: {max_eval}, Move: {move}")

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                #print("Beta cutoff (maximizing)")
                break  # Beta cut-off

        return max_eval, best_move
    else:
        min_eval = float('inf')
        moves = get_valid_moves(state, player)
        for move in moves:
            eval, _ = alpha_beta(move, depth + 1, alpha, beta, True, get_next_turn(player), depth_limit)
            #print(f"Eval: {eval}, Min Eval: {min_eval}, Move: {move}")

            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)

            if beta <= alpha:
                #print("Alpha cutoff (minimizing)")
                break  # Alpha cut-off
        return min_eval, best_move
    
def evaluate(state: State, depth: int, player: str) -> int:
    """
    Evaluate the board state based on the number of pieces, kings, positions, and potential future moves.
    The heuristic gives a higher score to positions favorable to 'player'.
    """
    # Simple heuristic based on the number of pieces and kings
    red_score = 0
    black_score = 0

    # Evaluate each red piece and give different scores for kings and normal pieces
    for pos, piece in state.red_pieces.items():
        if piece.is_king:
            red_score += 15  # Kings are more valuable
        else:
            red_score += 10  # Normal pieces are less valuable
            red_score += (7 - piece.row)  # Reward for getting closer to becoming a king (closer to row 0)

    # Evaluate each black piece similarly
    for pos, piece in state.black_pieces.items():
        if piece.is_king:
            black_score += 15  # Kings are more valuable
        else:
            black_score += 10  # Normal pieces are less valuable
            black_score += piece.row  # Reward for getting closer to becoming a king (closer to row 7)

    # The score difference between red and black
    score = red_score - black_score

    # Factor in game-over situations
    if game_over(state):
        winner = get_winner(state)
        if winner == 'r':
            return WIN_SCORE - depth  # Prioritize faster wins for red
        elif winner == 'b':
            return LOSS_SCORE + depth  # Slower losses are better for red
        else:
            return 0  # Draw

    # If the player is black, we negate the score to reflect that it's minimizing
    if player == 'b':
        score = -score

    #print(f"Evaluation Score: {score} at Depth: {depth} for Player: {player}")

    return score

def get_valid_moves(state: State, player: str) -> list:
    """
    Get all valid moves for the current player.
    """
    moves = []
    jump_moves = []
    for key in state.red_pieces.keys() if player == 'r' else state.black_pieces.keys():
            piece_moves, piece_jumps = get_piece_moves(state, key[0], key[1], state.red_pieces[key] if player == 'r' else state.black_pieces[key] , False)

            #print('Piece Moves')
            #print(piece_moves)

            if piece_jumps:
                jump_moves.extend(piece_jumps)
            else:
                moves.extend(piece_moves)

    if jump_moves:
        return jump_moves

    return moves

def get_piece_moves(state: State, row: int, col: int, piece: Piece, multi_jump: bool) -> list:
    """
    Get all valid moves for a piece at (row, col) for the current player.
    This function needs to handle both normal moves and jumps.
    """
    player = piece.player
    moves = []
    jumps = []


    if player in ['r', 'b']:
        directions = [(-1, -1), (-1, 1)] if player == 'r' else [(1, -1), (1, 1)]  # Red moves up, black moves down
    else:
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Kings move in all four diagonal directions + all 4 horizntal and vertical directions

    for dr, dc in directions:
        if multi_jump == False:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < state.height and 0 <= new_col < state.width:
                if state.empty_pieces.get((new_row, new_col)) != None:
                    # Simple move to an empty space

                    moves.append(create_state(state, ((row, col), (new_row, new_col), (0, 0))))

        # Check for jumps
        jump_row, jump_col = row + 2 * dr, col + 2 * dc
        mid_row, mid_col = row + dr, col + dc
        dict_curr = state.red_pieces if player == 'r' or player == 'R' else state.black_pieces
        dict_opp = state.black_pieces if player == 'r' or player == 'R' else state.red_pieces
        if (0 <= jump_row < state.height and 0 <= jump_col < state.width and
            dict_opp.get((mid_row, mid_col)) != None and state.empty_pieces.get((jump_row, jump_col)) != None):
            # Add jump move
            curr_jump_state = create_state(state, ((row, col), (jump_row, jump_col), (mid_row, mid_col)))
            dict_curr_jump = curr_jump_state.red_pieces if player == 'r' or player == 'R' else curr_jump_state.black_pieces

            # Check for multi-jumps
            jump_lst= get_piece_moves(curr_jump_state, jump_row, jump_col, dict_curr_jump.get((jump_row, jump_col)) , True)

            jumps.extend(jump_lst[1])

    if multi_jump == True and jumps == []:
        jumps.append(state)

    return (moves, jumps)

def create_state(state: State, move: tuple) -> State:
    """
    Apply the given move to the board. create a new state
    move: list of tuples of ((start_row, start_col), (end_row, end_col), (mid_row, mid_col))
    """

    #print('State')
    #print(state)

    #print('Move')
    #print(move)

    # Need to copy the state avoid deepcopy, just copy the board and create new pieces for the peices that moved 
    start, end, mid = move
    start_row, start_col = start
    end_row, end_col = end
    mid_row, mid_col = mid

    # Copy the board and dictionarys
    new_state = State(state.empty_pieces.copy(), state.red_pieces.copy(), state.black_pieces.copy())

    # Move the new piece into the spot
    dict_curr = new_state.red_pieces if new_state.red_pieces.get((start_row, start_col)) else new_state.black_pieces  
    other_dict = new_state.black_pieces if new_state.red_pieces.get((start_row, start_col)) else new_state.red_pieces
    
    # Normal move
    if mid_row == 0 and mid_col == 0:
            # Update new empty piece
            new_state.empty_pieces.pop((end_row, end_col))
            new_state.empty_pieces[(start_row, start_col)] = Piece('.', False, True, start_row, start_col)
            
            old_piece = dict_curr.pop((start_row, start_col))
            dict_curr[(end_row, end_col)] = Piece(old_piece.player, old_piece.is_king, False, end_row, end_col)

    # Jump move/moves 
    else:

        #print(start_row, start_col)
        # Pop both pieces
        old_piece = dict_curr.pop((start_row, start_col))
        old_piece_mid = other_dict.pop((mid_row, mid_col))

        # Update the empty pieces
        new_state.empty_pieces[(start_row, start_col)] = Piece('.', False, True, start_row, start_col)
        new_state.empty_pieces[(mid_row, mid_col)] = Piece('.', False, True, mid_row, mid_col) 
        
        # Update the final spot of the piece, pop the space that was in it before, then update the dictionary of the player
        new_state.empty_pieces.pop((end_row, end_col))
        dict_curr[(end_row, end_col)] = Piece(old_piece.player, old_piece.is_king, False, end_row, end_col)

    # Check for promotion to king
    if end_row == 0 and old_piece.player == 'r':
        dict_curr[(end_row, end_col)] = Piece('R', True , False, end_row, end_col)
    elif end_row == state.height - 1 and old_piece.player == 'b':
        dict_curr[(end_row, end_col)] = Piece('B', True , False, end_row, end_col)

    new_state.update_player_count()

    #print('New State')
    #print(new_state)

    return new_state

def game_over(state):
    """
    Finds if the game is over or not based on the current state.
    state: The current board state.
    """
    
    if (state.r_num == 0 and state.r_king == 0) or (state.b_num == 0 and state.b_king == 0):
        return True  # Game is over if any player has no pieces left
    
    if state.red_pieces == {} or state.black_pieces == {}: 
        return True

    # Check if either player has no valid moves
    if not get_valid_moves(state, 'r') or not get_valid_moves(state, 'b'):
        return True  # No valid moves for either player

    return False

def get_winner(state):
    """
    Returns the winning player ('r' or 'b') or None if no winner yet.
    state: The current board state.
    """
    # Returns the winning player ('r' or 'b') or None if no winner yet

    state.update_player_count()

    #print('Red Pieces')
    #print([str(state.red_pieces[key]) for key in list(state.red_pieces.keys())])
    #print('Black Pieces')
    #print([str(state.black_pieces[key]) for key in list(state.black_pieces.keys())])

    #print('Red Num')
    #print(state.r_num)
    #print(state.r_king)
    #print('Black Num')
    #print(state.b_num)
    #print(state.b_king)


    if state.r_num == 0 and state.r_king == 0:
        return 'b'  # Black wins
    if state.b_num == 0 and state.b_king == 0:
        return 'r'  # Red wins
    return None  # No winner yet



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    empty_pieces, red_pieces, black_pieces  = read_from_file(args.inputfile)
    state = State(empty_pieces, red_pieces, black_pieces)
    turn = 'r'
    move_number = 0  # Initialize move number

    #print(state)
    #print('Empty Pieces')
    #print([str(state.empty_pieces[key]) for key in list(state.empty_pieces.keys())])
    #print('Red Pieces')
    #print([str(state.red_pieces[key]) for key in list(state.red_pieces.keys())])
    #print('Black Pieces')
    #print([str(state.black_pieces[key]) for key in list(state.black_pieces.keys())])

    #print(state)
    #print(state.r_num)
    #print(state.b_num)
    #print(state.r_king)
    #print(state.b_king)

    sys.stdout = open(args.outputfile, 'w')
    print(state)
    
    # Create a stack to store the path to the solution 
    path = [state]

    # When solution found we will print each state of board to get to the solution
    while not game_over(state):
        # Find the best move for the current player using Alpha-Beta pruning
        depth_limit = adaptive_depth_limit(state, move_number)
        # the best move will be a ready to go state to use in the next iteration
        _, best_move = alpha_beta(state, 0, -float('inf'), float('inf'), True if turn == 'r' else False, turn, depth_limit)

        #print('Best Move')
        #print(best_move)
        # Apply the best move
        if best_move:
            path.append(best_move)
            state = best_move
            print(state)  # Print the new board state after each move

        # Increment move number after each valid move
        move_number += 1

        # Switch turns
        turn = get_next_turn(turn)

    sys.stdout = sys.__stdout__
