import time

import pytest
import pydin
from numba import jit


# Helper functions to judge tic-tac-toe game state
@jit(nopython=True)
def is_terminal(state):
    # Check rows
    for i in range(3):
        if abs(sum(state[i * 3:(i + 1) * 3])) == 3:
            return True
    # Check columns
    for i in range(3):
        if abs(sum(state[i::3])) == 3:
            return True
    # Check diagonals
    if abs(sum(state[::4])) == 3 or abs(sum(state[2:8:2])) == 3:
        return True
    return False


@jit(nopython=True)
def get_actions(state, _):
    # For Tic Tac Toe, the move doesn't affect which actions are possible
    # Therefore, we don't need to use it in this function.
    return [i for i in range(9) if state[i] == 0]


@jit(nopython=True)
def transition(state, action):
    new_state = state.copy()
    new_state[action] = 1
    return new_state


@jit(nopython=True)
def reward(state, is_terminal):
    if not is_terminal:
        return 0
    for i in range(3):
        if sum(state[i * 3:(i + 1) * 3]) == 3 or sum(state[i::3]) == 3:
            return 1
    if sum(state[::4]) == 3 or sum(state[2:8:2]) == 3:
        return 1
    return 0


#
# def test_mcts():
#     initial_state = [0] * 9
#     initial_action = -1
#
#     mcts = pydin.MCTS(
#         initial_state=initial_state,
#         initial_action=initial_action,
#         get_actions=get_actions,
#         transition=transition,
#         is_terminal=is_terminal,
#         reward=reward,
#     )
#
#     # Suppose your MCTS has a method called `best_action` that returns the best action.
#     # You can then assert that the best action is as expected. Below, we assume that
#     # the best action for the starting player in an empty tic-tac-toe grid is the middle square.
#     mcts.execute(10000)
#     # best_action = mcts.best_action()
#     time.sleep(1)
#     print("Hello World")
#     print("Best Action: ", mcts.best_action())
#     mcts.print_tree()
#     assert 4 == 4  # Middle square in 1D list
#
#
# if __name__ == "__main__":
#     import pytest
#
#     raise SystemExit(pytest.main([__file__]))


import pydin

initial_state = [0] * 9
initial_action = -1

mcts = pydin.MCTS(
    initial_state=initial_state,
    initial_action=initial_action,
    get_actions=get_actions,
    transition=transition,
    is_terminal=is_terminal,
    reward=reward)
