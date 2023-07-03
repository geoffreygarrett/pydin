load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "dist_info_requirement",
    "entry_point",
    "requirement",
)

py_test(
    name = "test_astrodynamics",
    srcs = ["test_astrodynamics.py"],
    args = [
        "--junit-xml=test_astrodynamics.xml",
    ],
    python_version = "PY3",
    deps = [
        "//pydin",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_mcts",
    srcs = ["test_mcts.py"],
    args = [
        "--junit-xml=test_mcts.xml",
    ],
    python_version = "PY3",
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("numba"),
    ],
)
