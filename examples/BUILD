load(
    "@pip_example//:requirements.bzl",
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
    name = "example_ellipsoidal_gravity_models",
    srcs = ["examples_gravity_models/ellipsoidal_gravity_models.py"],
    main = "examples_gravity_models/ellipsoidal_gravity_models.py",
    python_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("matplotlib"),
        "@//pydin/pydin",
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
        "//pydin/pydin",
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

## pybind11 pydin stubs #############################################################################
py_binary(
    name = "foo_test",
    srcs = ["examples_develop/foo_test.py"],
    data = [
        "//pydin/tests:data/Eros.face",
        "//pydin/tests:data/Eros.node",
        "@eros_50k_ply//file",
    ],
    legacy_create_init = False,
    visibility = ["//visibility:public"],
    deps = [
        "//pydin/pydin",
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("pillow"),
        requirement("sympy"),
        #requirement("pyvista"),

        # mayavi
        requirement("mayavi"),
        requirement("trimesh"),
        requirement("pyqt6"),
        requirement("pyside6"),
        #        entry_point("traitsui", "qt4"),
        #        requirement("wxPython"),
        #        requirement("pyside2"),
        #        requirement("pyside2"),
        requirement("matplotlib"),
        #        entry_point("setuptools", "easy_install"),
        # @pip_setuptools
    ],
)

py_binary(
    name = "dev",
    srcs = ["examples_develop/dev.py"],
    data = [
        "//pydin/tests:data/Eros.face",
        "//pydin/tests:data/Eros.node",
        "@eros_50k_ply//file",
    ],
    legacy_create_init = False,
    visibility = ["//visibility:public"],
    deps = [
        "@@bor//pydin/pydin",
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("pillow"),
        requirement("sympy"),
        #requirement("pyvista"),

        # mayavi
        requirement("mayavi"),
        requirement("trimesh"),
        requirement("pyqt6"),
        requirement("pyside6"),
        #        entry_point("traitsui", "qt4"),
        #        requirement("wxPython"),
        #        requirement("pyside2"),
        #        requirement("pyside2"),
        requirement("matplotlib"),
        #        entry_point("setuptools", "easy_install"),
        # @pip_setuptools
    ],
)

py_binary(
    name = "example_autodiff_stts",
    srcs = ["example_autodiff_stts.py"],
    deps = [
        requirement("sympy"),
        requirement("numpy"),
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "example_autodiff_stts2",
    srcs = ["example_autodiff_stts2.py"],
    deps = [
        requirement("sympy"),
        requirement("numpy"),
        requirement("matplotlib"),
    ],
)
