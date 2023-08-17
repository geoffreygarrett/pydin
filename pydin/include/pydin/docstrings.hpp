const std::string mcts_docstrings = R"doc(
    Construct a new MCTS object

    Parameters
    ----------
    initial_state : State
        The initial state of the MCTS.
    initial_action : Action
        The initial action of the MCTS.
    action_generator : function
        A function that generates the possible actions for a given state.
    transition : function
        A function that takes a state and action, and returns a new state.
    is_terminal : function
        A function that checks whether a given state is terminal.
    reward : function
        A function that calculates the reward for a given state.
    selection_policy : function, optional
        A function that determines the selection policy, by default UBC1.
    seed : int, optional
        A seed for the random number generator, by default random.

    Returns
    -------
    MCTS
        An initialized MCTS object.
    )doc";

const std::string mcts_repr_docstrings = R"doc(
    Convert the MCTS object to a string representation.

    Returns
    -------
    str
        The string representation of the MCTS object.
    )doc";

const std::string mcts_search_docstrings = R"doc(
    Perform a search with the given number of iterations.

    Parameters
    ----------
    iterations : int, optional
        The number of iterations for the search, by default 1000.
    seconds : float, optional
        The number of seconds for the search, by default -1.0.
    expand_all : bool, optional
        Whether to expand all nodes, by default False.

    Returns
    -------
    None
    )doc";


const std::string mcts_get_root_docstrings = R"doc(
    Get the root node of the tree.

    Returns
    -------
    Node
        The root of the MCTS.
    )doc";