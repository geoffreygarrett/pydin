import numba as nb
import numpy as np

import pydin

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
    if (np.abs(np.diag(state_grid).sum()) == DIM) | (
            np.abs(np.diag(state_grid[::-1]).sum()) == DIM
    ):
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
def transition_fn(state: np.ndarray[np.int32], action: int) -> np.ndarray[np.int32]:
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


import pydin.core.logging as pdlog
import concurrent.futures


# Define a function that runs MCTS and returns the result
def run_mcts(i: int):
    pdlog.info("Starting MCTS {}".format(i))
    initial_state = np.concatenate((np.array([0] * (DIM ** 2)), np.array([1])), axis=0)
    iterations = 1000
    try:
        pdlog.debug("Starting MCTS 2222")
        mcts = pydin.MCTS(
            initial_state=initial_state,
            action_generator=get_actions_fn,
            transition=transition_fn,
            is_terminal=is_terminal_fn,
            reward=reward_fn,
            selection_policy=pydin.EpsilonGreedy(epsilon=0.01),
        )

        trajectory, metrics = mcts.search(
            iterations=iterations,
            seconds=-1.0,
            expand_all=False,
            contraction=True,
            leaf_parallelism=4,
        )
        return trajectory, metrics
    except Exception as e:
        print("Exception occurred: ", e)
        raise


def main2():
    pdlog.info("Starting MCTS")
    pdlog.set_level(pdlog.DEBUG)

    # Define the number of MCTS runs
    num_mcts_runs = 4

    # Run MCTS in parallel using concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Run num_mcts_runs MCTS in parallel
        results = executor.map(run_mcts, range(num_mcts_runs))

    return results


def main1():
    # // change log pattern
    # pdlog.set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v")
    pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting MCTS")
    initial_state = np.concatenate((np.array([0] * (DIM ** 2)), np.array([1])), axis=0)

    # root = pydin.PyNode(state=initial_state)
    # mcts = pydin.MCTS(root=root)
    # mcts.set_fn_get_actions(get_actions_fn)
    # mcts.set_fn_is_terminal(is_terminal_fn)
    # mcts.set_fn_reward(reward_fn)
    # mcts.set_fn_transition(transition_fn)
    # mcts.set_fn_selection_policy(pydin.EpsilonGreedy(epsilon=0.01))

    mcts = pydin.MCTS(
        initial_state=initial_state,
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
    iterations = 100
    timer_name = "MCTS Search, {} iterations".format(iterations)
    pdlog.start_timer(timer_name)

    trajectory, metrics = mcts.search(
        iterations=iterations,
        seconds=-1.0,
        expand_all=False,
        contraction=True,
        leaf_parallelism=4,
    )

    # root = mcts.get_root()
    # print(root)
    # root.to_binary("root.bin")
    # root = pydin.PyNode.load_binary("root.bin")
    # print(root)

    # # return
    # print(mcts.get_root().get_children())
    # print(mcts.get_root().get_children()[0].state)

    print(mcts.get_root())
    # mcts.to_json("mcts_pre.json")
    # mcts = pydin.MCTS.load_json("mcts_pre.json")
    # print(mcts.get_root())

    # return 0
    # mcts.to_binary("mcts_pre.bin")
    # mcts = pydin.MCTS.load_binary("mcts_pre.bin")
    # print(mcts)

    # mcts.set_fn_get_actions(get_actions_fn)
    # mcts.set_fn_is_terminal(is_terminal_fn)
    # mcts.set_fn_reward(reward_fn)
    # mcts.set_fn_transition(transition_fn)
    # mcts.set_fn_selection_policy(pydin.EpsilonGreedy(epsilon=0.01))

    # print(mcts.get_root().get_children())
    # print(mcts.get_root().get_children()[0].state)

    # mcts.to_binary("mcts_post.bin")
    # mcts = pydin.MCTS.load_binary("mcts_post.bin")
    #
    pdlog.stop_timer(timer_name)
    parent = mcts.get_root()
    #
    # mcts.to_json("mcts.json")
    # mcts.to_binary("mcts.bin")
    #
    # mcts1 = pydin.MCTS.load_json("mcts.json")
    # mcts = pydin.MCTS.load_binary("mcts.bin")

    # parent = mcts.get_root()
    l = parent.get_children()
    for _l in l:
        print("Child: ", _l)
        print("Childs Parent: ", _l.get_parent())

    print("Parent: ", parent)
    parent.to_json("parent.json")
    parent.to_binary("parent.bin")

    # test = pydin.PyNode.load_json("parent.json")
    test = pydin.PyNode.load_binary("parent.bin")
    print("Test: ", test)
    l = test.get_children()
    for _l in l:
        print("Child: ", _l)

    import pickle

    # save pickle
    with open("parent.pkl", "wb") as f:
        pickle.dump(parent, f)

    # load pickle and test
    with open("parent.pkl", "rb") as f:
        loaded_obj = pickle.load(f)

    print("Parent: ", loaded_obj)
    l = parent.get_children()
    for _l in l:
        print("Child: ", _l)
        print("Childs Parent: ", _l.get_parent())

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


if __name__ == "__main__":
    # results = main2()
    main1()
    # for idx, result in enumerate(results):
    #     trajectory, metrics = result
    #     print("Trajectory: ", trajectory)
    #     print("Metrics: ", metrics)
    #     metrics.to_json("metrics_{}.json".format(idx))
    #
    #     # test "from_json"
    #     metrics2 = pydin.SearchMetrics.load_json("metrics_{}.json".format(idx))
    #     print("Metrics2: ", metrics2)
