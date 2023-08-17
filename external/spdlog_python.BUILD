load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "filter_directory", "pkg_files", "pkg_mklink", "strip_prefix")
load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

pybind_extension(
    name = "spdlog",
    srcs = ["src/pyspdlog.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gabime_spdlog//:spdlog",
    ],
)

alias(
    name = "black",
    actual = entry_point("black", "black"),
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

py_binary(
    name = "stub_generator",
    srcs = ["//:tools/stub_generator.py"],
    deps = [
        requirement("pybind11-stubgen"),
        requirement("black"),
    ],
)

pybind_stubgen(
    name = "spdlog_stubs",
    src = ":spdlog",
    code_formatter = ":black",
    ignore_invalid = [
        "signature",
        "defaultarg",
    ],
    log_level = "DEBUG",
    module_name = "spdlog",
    no_setup_py = True,
    root_module_suffix = "",
    tool = ":stub_generator",
)

filegroup(
    name = "py.files",
    srcs = glob([
        "**/*.py",
        "__init__.py",
    ]),
)

py_library(
    name = "spdlog",
    srcs = [
        ":py.files",
    ],
    data = [
        ":__init__",
        ":core",
        ":core_stubs",
    ],
    imports = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        #        requirement("spdlog"),
    ],
)
