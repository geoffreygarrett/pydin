load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "filter_directory", "pkg_files", "pkg_mklink", "strip_prefix")
load(
    "@pip_build//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

alias(
    name = "pydin",
    actual = "@pydin//:pydin",
)

## PYBIND EXTENSION: CORE ###############################################
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

## pydin library ####################################################################################
alias(
    name = "black",
    actual = entry_point("black", "black"),
)

exports_files(["tools/stub_generator.py"])

py_binary(
    name = "stub_generator",
    srcs = ["tools/stub_generator.py"],
    deps = [],
)

genrule(
    name = "__init__",
    outs = ["__dummy__.py"],
    cmd = """
    OUT_FILE=$$(mktemp)
    echo "" > $${OUT_FILE}
    mv $${OUT_FILE} $(OUTS)
    """,
    tools = [
        ":black",
        ":stub_generator",
    ],
)

genrule(
    name = "README",
    outs = ["README.md"],
    cmd = """
    OUT_FILE=$$(mktemp)
    echo "# pydin" > $${OUT_FILE}
    echo "Python bindings for the Odin library" >> $${OUT_FILE}
    echo "" >> $${OUT_FILE}
    echo "## Dependency Installation" >> $${OUT_FILE}
    echo "\\`\\`\\`bash" >> $${OUT_FILE}
    echo "pip install -r requirements.txt" >> $${OUT_FILE}
    echo "\\`\\`\\`" >> $${OUT_FILE}
    echo "## Testing" >> $${OUT_FILE}
    echo "\\`\\`\\`bash" >> $${OUT_FILE}
    echo "python -m pytest" >> $${OUT_FILE}
    echo "\\`\\`\\`" >> $${OUT_FILE}
    echo "## Examples" >> $${OUT_FILE}
    echo "\\`\\`\\`bash" >> $${OUT_FILE}
    echo "python -m examples.<example_filename>" >> $${OUT_FILE}
    echo "\\`\\`\\`" >> $${OUT_FILE}
    echo "" >> $${OUT_FILE}
    mv $${OUT_FILE} $(OUTS)
    """,
    tools = [
        ":black",
        ":stub_generator",
    ],
)

pkg_files(
    name = "pydin_module",
    srcs = [
        ":__init__",
        "//pydin:core_stubs",
        "//pydin:py.files",
        "//pydin:py.typed",
        "//pydin/pydin:core",
    ],
    prefix = "pydin",
    visibility = ["//visibility:public"],
)

pkg_files(
    name = "pydin_tests",
    srcs = ["//pydin/tests:test_files"],
    prefix = "tests",
    visibility = ["//visibility:private"],
)

pkg_files(
    name = "pydin_examples",
    srcs = ["//pydin/examples:example_files"],
    prefix = "examples",
    visibility = ["//visibility:private"],
)

pkg_files(
    name = "environment_files",
    srcs = [
        "//pydin:environment.yml",
        "//pydin:requirements.txt",
    ],
    visibility = ["//visibility:private"],
)

PKG_SRCS = [
    ":README",
    ":environment_files",
    ":pydin_examples",
    ":pydin_module",
    ":pydin_tests",
]

PKG_STAMP = 1

PKG_STRIP_PREFIX = strip_prefix.from_pkg("external/pydin")

pkg_zip(
    name = "pydin-zip",
    srcs = PKG_SRCS,
    out = "pydin.zip",
    stamp = PKG_STAMP,
    strip_prefix = PKG_STRIP_PREFIX,
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "pydin-tar",
    srcs = PKG_SRCS,
    out = "pydin.tar.gz",
    stamp = PKG_STAMP,
    strip_prefix = PKG_STRIP_PREFIX,
    visibility = ["//visibility:public"],
)
