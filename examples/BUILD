load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "dist_info_requirement",
    "entry_point",
    "requirement",
)

filegroup(
    name = "example_files",
    srcs = [
        "model_gravitation_ellipsoidal.py",
    ],
    visibility = ["//visibility:public"],
)

#py_binary(
#    name = "orbit_plot",
#    srcs = ["orbit_plot.py"],
#    python_version = "PY3",
#    deps = [
#        requirement("numpy"),
#        requirement("matplotlib"),
#        "@//pydin",
#    ],
#)

py_binary(
    name = "model_gravitation_ellipsoidal",
    srcs = ["model_gravitation_ellipsoidal.py"],
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
        "@//pydin",
    ],
)

py_binary(
    name = "odinscript",
    srcs = ["odinscript.py"],
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
        "@//pydin",
    ],
)

py_binary(
    name = "basic_triaxial_ellipsoid_example",
    srcs = ["basic_triaxial_ellipsoid_example.py"],
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
        "@//pydin",
    ],
)

py_binary(
    name = "advanced_triaxial_ellipsoid_tbb_example",
    srcs = ["advanced_triaxial_ellipsoid_tbb_example.py"],
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
        "@//pydin",
    ],
)

#py_binary(
#    name = "main",
#    srcs = ["main.py"],
#    python_version = "PY3",
#    deps = [
#        requirement("numpy"),
#        requirement("matplotlib"),
#        requirement("numba"),
#        "@//pydin",
#    ],
#)

#py_binary(
#    name = "a_preliminary",
#    srcs = [
#        "gtoc12/a_preliminary.py",
#        "gtoc12/util.py",
#    ],
#    data = ["//pydin/examples/data:GTOC12_Asteroids_Data.txt"],
#    python_version = "PY3",
#    deps = [
#        requirement("numpy"),
#        requirement("matplotlib"),
#        requirement("scikit-learn"),
#        "@//pydin",
#    ],
#)

#py_binary(
#    name = "gtoc_clustering",
#    srcs = ["gtoc_clustering.py"],
#    #    data = ["//pydin/examples/data:GTOC12_Asteroids_Data.txt"],
#    python_version = "PY3",
#    deps = [
#        requirement("numpy"),
#        requirement("matplotlib"),
#        requirement("pandas"),
#        requirement("scikit-learn"),
#        "@//pydin",
#    ],
#)

py_binary(
    name = "mcts_tictactoe",
    srcs = ["mcts_tictactoe.py"],
    python_version = "PY3",
    deps = [
        requirement("numba"),
        "@//pydin",
    ],
)
#
#py_binary(
#    name = "mcts",
#    srcs = ["mcts.py"],
#    #    data = ["//pydin/examples/data:GTOC12_Asteroids_Data.txt"],
#    python_version = "PY3",
#    deps = [
#        #        requirement("numpy"),
#        #        requirement("matplotlib"),
#        requirement("numba"),
#        requirement("python-igraph"),
#        requirement("plotly"),
#        #        requirement("scikit-learn"),
#        "@//pydin",
#    ],
#)
