load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "dist_info_requirement",
    "entry_point",
    "requirement",
)

exports_files([
    "data/Eros.face",
    "data/Eros.node",
])

filegroup(
    name = "test_files",
    srcs = [
        "test_astrodynamics.py",
        "test_mcts.py",
    ],
    visibility = ["//visibility:public"],
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
        "//pydin",
        requirement("pytest"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_tree",
    srcs = ["test_tree.py"],
    args = [
        "--junit-xml=test_tree.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_geometry",
    srcs = ["test_geometry.py"],
    args = [
        "--junit-xml=test_geometry.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_shape",
    srcs = ["test_shape.py"],
    args = [
        "--junit-xml=test_tree.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_plotting",
    srcs = ["test_plotting.py"],
    args = [
        "--junit-xml=test_plotting.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("pillow"),
        requirement("pyvista"),

        # mayavi
        requirement("mayavi"),
        requirement("trimesh"),
        requirement("PySide2"),
        requirement("matplotlib"),
    ],
)

py_test(
    name = "test_gravitation",
    srcs = ["test_gravitation.py"],
    args = [
        "--junit-xml=test_astrodynamics.xml",
        "-v",
    ],
    data = [
        "Eros.face",
        "Eros.node",
        "@eros_50k_ply//file",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
        requirement("pandas"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_tbb",
    srcs = ["test_tbb.py"],
    args = [
        "--junit-xml=test_astrodynamics.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_linalg",
    srcs = ["test_linalg.py"],
    args = [
        "--junit-xml=test_linalg.xml",
        "-v",
    ],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//pydin",
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
        "//pydin",
        requirement("pytest"),
        requirement("numba"),
    ],
)
