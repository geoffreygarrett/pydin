import pydin
import numba as nb
import numpy as np

DIM = 5


@nb.njit(nogil=True)
def is_terminal_fn(state: np.ndarray[int]) -> bool:
    state_grid = state.reshape((DIM, DIM))

    # Check rows and columns
    for i in range(DIM):
        if np.abs(state_grid[i, :].sum()) == DIM or np.abs(state_grid[:, i].sum()) == DIM:
            return True

    # Check diagonals
    if ((np.abs(np.diag(state_grid).sum()) == DIM)
            |
            (np.abs(np.diag(state_grid).sum()) == DIM)):
        return True

    # Otherwise, not terminal
    return False


@nb.njit(nogil=True)
def get_actions_fn(state: np.ndarray[np.int32]) -> nb.typed.List[np.int32]:
    return np.where(state == 0)[0]  # Available action, wherever there is a 0


@nb.njit(nogil=True)
def transition_fn(state: np.ndarray[np.int32], action: np.int32) -> np.ndarray[np.int32]:
    new_state = np.copy(state)
    new_state[action] = np.int32(1)
    return new_state


@nb.njit(nogil=True)
def reward_fn(state: np.ndarray[np.int32], is_terminal: bool) -> float:
    state_grid = state.reshape((DIM, DIM))

    if not is_terminal:
        return 0

    # Check rows and columns
    for i in range(DIM):
        if np.abs(state_grid[i, :].sum()) == DIM or np.abs(state_grid[:, i].sum()) == DIM:
            return 1

    # Check diagonals
    if ((np.abs(np.diag(state_grid).sum()) == DIM)
            |
            (np.abs(np.diag(state_grid).sum()) == DIM)):
        return 1

    return 0


if __name__ == "__main__":
    initial_state = np.array([0] * (DIM ** 2))  # Use Numba's typed list
    initial_action = -1

    mcts = pydin.MCTS(
        initial_state=initial_state,
        initial_action=initial_action,
        get_actions=get_actions_fn,
        transition=transition_fn,
        is_terminal=is_terminal_fn,
        reward=reward_fn,
    )

    # Suppose your MCTS has a method called `best_action` that returns the best action.
    # You can then assert that the best action is as expected. Below, we assume that
    # the best action for the starting player in an empty tic-tac-toe grid is the middle square.
    mcts.execute(100000)
    # best_action = mcts.best_action()
    mcts.print_tree(-1, True)
    # print("Best Action: ", mcts.best_action())

    # assert 4 == 4  # Middle square in 1D list
