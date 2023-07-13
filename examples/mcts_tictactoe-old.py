import numba as nb
import numpy as np
from numpy.typing import NDArray

import pydin

DIM: np.int32 = np.int32(3)


# @nb.cfunc("int64(int64[:])", nogil=True)
# def check_winner(state: NDArray[np.int32]) -> np.int64:
#     state_grid = state[:-1].reshape((DIM, DIM))
#
#     for i in range(DIM):
#         if np.abs(state_grid[i, :].sum()) == DIM:
#             return np.sign(state_grid[i, :].sum())
#         elif np.abs(state_grid[:, i].sum()) == DIM:
#             return np.sign(state_grid[:, i].sum())
#
#     # Check diagonals
#     if ((np.abs(np.diag(state_grid).sum()) == DIM)
#             |
#             (np.abs(np.diag(state_grid[::-1]).sum()) == DIM)):
#         return np.sign(np.diag(state_grid).sum())
#
#     # No winner
#     return 0
@nb.cfunc("int32(int32[:])", nogil=True)
def check_winner(state: nb.types.Array) -> nb.int32:
    # Calculate the size of the state grid
    grid_size = int(np.sqrt(state.size - 1))

    # Process the state grid manually
    for i in range(grid_size):
        row_sum = 0
        col_sum = 0
        for j in range(grid_size):
            row_sum += state[i * grid_size + j]
            col_sum += state[j * grid_size + i]
        if np.abs(row_sum) == grid_size or np.abs(col_sum) == grid_size:
            return np.sign(row_sum)

    # Check diagonals
    diag_sum1 = 0
    diag_sum2 = 0
    for i in range(grid_size):
        diag_sum1 += state[i * grid_size + i]
        diag_sum2 += state[i * grid_size + (grid_size - i - 1)]
    if np.abs(diag_sum1) == grid_size or np.abs(diag_sum2) == grid_size:
        return np.sign(diag_sum1)

    # No winner
    return 0


@nb.cfunc("boolean(int32[:])", nogil=True)
def contains_zero(state: NDArray[np.int32]) -> bool:
    for i in range(state.size - 1):  # Exclude the last element
        if state[i] == 0:
            return True
    return False


@nb.cfunc("boolean(int32[:])", nogil=True)
def is_terminal_fn(state: NDArray[np.int32]) -> bool:
    if check_winner(state) != 0 or not contains_zero(state):
        return True

    return False


@nb.cfunc("int64[:](int64[:])", nogil=True)
def get_actions_fn(state: NDArray[np.int32]) -> nb.int64[:]:
    return np.where(state == 0)[0]


@nb.cfunc("int64[:](int64[:], int64)", nogil=True)
def transition_fn(state: nb.int64[:],
                  action: nb.int64) -> nb.int64[:]:
    new_state = np.copy(state)
    new_state[action] = new_state[-1]  # Set the value to the current player's move
    new_state[-1] = -new_state[-1]  # Switch to the other player
    return new_state


@nb.cfunc("float32(int32[:])", nogil=True)
def reward_fn(state: NDArray[np.int32]) -> float:
    winner: np.int32 = check_winner(state)

    # Count the number of moves taken
    num_moves = np.count_nonzero(state[:-1])

    # Add a small number to avoid division by zero
    num_moves = max(num_moves, 1)

    if winner == 1:
        return 1.0 / num_moves
    elif winner == -1:
        return 0.0
    elif is_terminal_fn(state):
        return 0.0
        # return 0.5 / num_moves  # the game is a draw

    return 0.0  # the game is not finished yet


import pydin.core.logging as pdlog

if __name__ == "__main__":
    # pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting MCTS")
    # pydin.config.set_log_verbosity(60)
    initial_state = np.concatenate((np.array([0] * (DIM ** 2)), np.array([1])), axis=0)
    initial_action = -1

    mcts = pydin.MCTS(
        initial_state=initial_state,
        initial_action=initial_action,
        action_generator=get_actions_fn,
        transition=transition_fn,
        is_terminal=is_terminal_fn,
        reward=reward_fn,
        selection_policy=pydin.EpsilonGreedy(epsilon=0.01),
        # selection_policy=pydin.UCB1(0.01),
    )

    # Suppose your MCTS has a method called `best_action` that returns the best action.
    # You can then assert that the best action is as expected. Below, we assume that
    # the best action for the starting player in an empty tic-tac-toe grid is the middle square.
    iterations = 1000
    timer_name = "MCTS Search, {} iterations".format(iterations)
    pdlog.start_timer(timer_name)
    trajectory, metrics = mcts.search(iterations=iterations, leaf_parallelism=1)
    pdlog.stop_timer(timer_name)

    print(metrics)
    # convert to state, action, reward
    # s, a, r = zip(*trajectory)

    # Let's print out the actions in the trajectory
    for state, action, reward in trajectory:
        # print with X and O
        state = state[:-1].reshape(DIM, DIM)
        new_state = np.copy(state).astype(str)
        new_state[state == 1] = "X"
        new_state[state == -1] = "O"
        new_state[state == 0] = "."
        print(f"State: \n{new_state}, \nAction: {action}, \nReward: {reward}")

    print("complete")
