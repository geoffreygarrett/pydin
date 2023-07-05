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
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:pydin",
        requirement("pytest"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_mcts",
    srcs = ["test_mcts.py"],
    args = [
        "--junit-xml=test_mcts.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:pydin",
        requirement("pytest"),
        requirement("numba"),
    ],
)
