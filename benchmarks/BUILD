# Load the necessary rules from the pip repository
load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "dist_info_requirement",
    "entry_point",
    "requirement",
)

# Define a group of source files which include all .py files
filegroup(
    name = "bechmark_files",
    srcs = glob(["*.py"]),
    # Allow access from any package
    visibility = ["//visibility:public"],
)

# Define a python library for utility functions used in benchmarks
py_library(
    name = "bench",
    srcs = [
        #        ":bench/__init__.py",
        "__init__.py",
        #        ":bench/utils.py",
        "utils.py",
    ],
    imports = [
    ],
    # Allow access from any package
    visibility = ["//visibility:public"],
    deps = [
        # Dependencies for this library include matplotlib and numpy
        requirement("matplotlib"),
        requirement("numpy"),
        "//pydin",
    ],
)

py_test(
    name = "test_benchmark",
    srcs = ["test_benchmark.py"],
    deps = [
        requirement("pytest"),
        ":bench",
    ],
)

# Define a python binary for the linear algebra benchmark
py_binary(
    name = "benchmark_linalg",
    srcs = ["benchmark_linalg.py"],
    args = [
        # Arguments specify the output xml file and verbose output
        "--junit-xml=benchmark_linalg.xml",
        "-v",
    ],
    # Allow access from any package
    imports = [
        #        ".",
        #        "bench",
    ],
    # Specify Python 3 as the version to use
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # Dependencies for this binary include the benchmark_utils library and the pydin package
        "//pydin",
        "//benchmarks:bench",
        requirement("seaborn"),
        requirement("pandas"),
        requirement("numba"),
    ],
)

# Define a python binary for the gravitation benchmark
py_binary(
    name = "benchmark_gravitation",
    srcs = ["benchmark_gravitation.py"],
    args = [
        # Arguments specify the output xml file and verbose output
        "--junit-xml=benchmark_gravitation.xml",
        "-v",
    ],
    # Specify Python 3 as the version to use
    python_version = "PY3",
    # Allow access from any package
    visibility = ["//visibility:public"],
    deps = [
        # Dependencies for this binary include the benchmark_utils library and the pydin package
        "//benchmarks:bench",
        requirement("seaborn"),
        requirement("pandas"),
        requirement("numba"),
        requirement("trimesh"),
        "//pydin",
    ],
)
