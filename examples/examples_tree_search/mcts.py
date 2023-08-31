import pydin
import numba as nb
import numpy as np
from typing import Tuple

DIM: np.int32 = np.int32(3)


@nb.njit(nogil=True)
def check_winner(state: np.ndarray[np.int32]) -> np.int32:
    state_grid = state[:-1].reshape((DIM, DIM))

    for i in range(DIM):
        if np.abs(state_grid[i, :].sum()) == DIM:
            return np.sign(state_grid[i, :].sum())
        elif np.abs(state_grid[:, i].sum()) == DIM:
            return np.sign(state_grid[:, i].sum())

    # Check diagonals
    if ((np.abs(np.diag(state_grid).sum()) == DIM)
            |
            (np.abs(np.diag(state_grid[::-1]).sum()) == DIM)):
        return np.sign(np.diag(state_grid).sum())

    # No winner
    return 0


@nb.njit(nogil=True)
def contains_zero(state: np.ndarray[np.int32]) -> bool:
    for i in range(state.size - 1):  # Exclude the last element
        if state[i] == 0:
            return True
    return False


@nb.njit(nogil=True)
def is_terminal_fn(state: np.ndarray[np.int32]) -> bool:
    if check_winner(state) != 0 or not contains_zero(state):
        return True

    return False


@nb.njit(nogil=True)
def get_actions_fn(state: np.ndarray[np.int32]) -> nb.typed.List[np.int32]:
    return np.where(state[:-1] == 0)[0]  # Available action, wherever there is a 0


@nb.njit(nogil=True)
def transition_fn(state: np.ndarray[np.int32],
                  action: np.int32) -> np.ndarray[np.int32]:
    new_state = np.copy(state)
    new_state[action] = new_state[-1]  # Set the value to the current player's move
    new_state[-1] = -new_state[-1]  # Switch to the other player
    return new_state


@nb.njit(nogil=True)
def reward_fn(state: np.ndarray[np.int32]) -> float:
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


if __name__ == "__main__":
    # pydin.config.set_log_verbosity(60)
    pydin.logging.log_info("Starting MCTS")
    initial_state = np.concatenate((np.array([0] * (DIM ** 2)), np.array([1])), axis=0)
    initial_action = -1
    # selection_policy = pydin.UCB1(0.01)
    selection_policy = pydin.EpsilonGreedy(epsilon=0.01)
    # Use the max as the value estimator
    # max_estimator = pydin.UCB1(1.0, lambda node: node.reward_stats.max())

    # Use the min as the value estimator
    # selection_policy = pydin.EpsilonGreedy(0.01, lambda node: node.reward_stats.max())

    mcts = pydin.MCTS(
        initial_state=initial_state,
        initial_action=initial_action,
        action_generator=get_actions_fn,
        transition=transition_fn,
        is_terminal=is_terminal_fn,
        reward=reward_fn,
        selection_policy=selection_policy
    )

    # Suppose your MCTS has a method called `best_action` that returns the best action.
    # You can then assert that the best action is as expected. Below, we assume that
    # the best action for the starting player in an empty tic-tac-toe grid is the middle square.
    trajectory = mcts.search(iterations=int(2e6), seconds=22.)
    # convert to state, action, reward
    s, a, r = zip(*trajectory)
    print(mcts)

    # def print_tree(node, level=0):
    #     print(' ' * level * 4, node)  # Adjust the number '4' to set the desired indentation.
    #     for child in node.get_children():
    #         print_tree(child, level + 1)

    root = mcts.get_root()
    # print_tree(root)
    # print(mcts)

    # Let's print out the actions in the trajectory
    for state, action, reward in trajectory:
        # print with X and O
        state = state[:-1].reshape(DIM, DIM)
        new_state = np.copy(state).astype(str)
        new_state[state == 1] = "X"
        new_state[state == -1] = "O"
        new_state[state == 0] = "."
        print(f"State: \n{new_state}, \nAction: {action}, \nReward: {reward}")

    pydin.logging.log_info("Finished MCTS")
