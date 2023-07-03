import pydin
import numba as nb
import numpy as np
from typing import Tuple

DIM: np.int32 = np.int32(3)


@nb.njit(nogil=True)
def check_winner(state: np.ndarray[np.int32]) -> np.int32:
    state_grid = state.reshape((DIM, DIM))

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
    for i in range(state.size):
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
    return np.where(state == 0)[0]  # Available action, wherever there is a 0


@nb.njit(nogil=True)
def transition_fn(state: np.ndarray[np.int32],
                  action: np.int32) -> Tuple[np.ndarray[np.int32], np.int32]:
    new_state = np.copy(state)
    # new_state[action] = move
    # new_move = -move  # Assuming move can only be 1 or -1, switch to the other player
    new_state[action] = np.int32(1)
    # return new_state, new_move
    return new_state


@nb.njit(nogil=True)
def reward_fn(state: np.ndarray[np.int32]) -> float:
    winner: np.int32 = check_winner(state)

    if winner == 1:
        return 1.0
    elif winner == -1:
        return -1.0
    elif is_terminal_fn(state):
        return 0.5  # the game is a draw

    return 0  # the game is not finished yet


if __name__ == "__main__":
    initial_state = np.array([0] * (DIM ** 2))  # Use Numba's typed list
    initial_action = -1

    mcts = pydin.MCTS(
        initial_state=initial_state,
        initial_action=initial_action,
        action_generator=get_actions_fn,
        transition=transition_fn,
        is_terminal=is_terminal_fn,
        reward=reward_fn,
        selection_policy=pydin.UB1(1.0),
    )

    # Suppose your MCTS has a method called `best_action` that returns the best action.
    # You can then assert that the best action is as expected. Below, we assume that
    # the best action for the starting player in an empty tic-tac-toe grid is the middle square.
    trajectory = mcts.search(iterations=10000)
    # convert to state, action, reward
    s, a, r = zip(*trajectory)
    # print(state)
    # print(action)
    # print(reward)

    print(mcts)

    root = mcts.get_root()
    # print(root)
    # print(root.state)
    # print(root.action)

    children = root.get_children()
    # print(children)

    # for child in children:
    #     print(child)
    #     print(child.state)
    #     print(child.action)
    #     print(child.reward)
    #     print(child.get_children())

    # raise Exception("Done")
    # best_action = mcts.best_action()
    # mcts.print_tree(-1, True)
    # print("Best Action: ", mcts.best_action())

    # assert 4 == 4  # Middle square in 1D list
    print("complete")
