load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "filter_directory", "pkg_files", "pkg_mklink", "strip_prefix")
load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)

#string_flag(
#    name = "gsl_library",
#    default = "enabled",
#    doc = "Configuration to use gsl library",
#    values = [
#        "enabled",
#        "disabled",
#    ],
#)

#config_setting(
#    name = "without_gsl",
#    flag_values = {gsl_library: "disabled"},
#)

# without gsl for windows and mac
#config_setting(
#    name = "without_gsl",
#    flag_values = {
#        "@platforms//os:windows": "disabled",
#        "@platforms//os:osx": "disabled",
#    },
#)

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

## PYBIND EXTENSION: CORE ###############################################
config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
)

CORE_LINKOPTS = select({
    ":msvc_compiler": [
        #        "/openmp",
    ],
    "@platforms//os:osx": [
        # openmp
        #        "-lomp",
        #        "-fopenmp",
    ],
    "//conditions:default": [
        #        "-fopenmp",
    ],
})

CORE_COPTS = select({
    ":msvc_compiler": [
        #        "/openmp",
    ],
    "@platforms//os:osx": [

        # openmp
        "-Xpreprocessor",
        #        "-fopenmp",
    ],
    "//conditions:default": [
        #        "-fopenmp",
    ],
})

filegroup(
    name = "core_hdrs",
    srcs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
        "include/**/*.cuh",
    ]),
)

pybind_library(
    name = "core_libs",
    hdrs = [":core_hdrs"],
    copts = CORE_COPTS,
    defines = [

        #        "PYBIND11_DETAILED_ERROR_MESSAGES",
        #        "ODIN_USE_GLOG",
        #        "GLOG_CUSTOM_PREFIX_SUPPORT",
    ],
    includes = ["include"],
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        #        "@com_github_google_glog//:glog",
        "@com_github_oneapi_onetbb//:tbb",
        "@com_github_uscilab_cereal//:cereal",
        "@odin",
    ] + select({
        "@platforms//os:osx": [
            #            "@org_gnu_gsl//:gsl",
        ],
        "@platforms//os:windows": [
            #            "@org_gnu_gsl//:gsl",
        ],
        "//conditions:default": [
            "@org_gnu_gsl//:gsl",
        ],
    }),
)

pybind_extension(
    name = "core",
    srcs = ["src/core.cpp"],
    copts = CORE_COPTS,
    defines = select({
        "@platforms//os:osx": [
            #            "ODIN_USE_GSL",
        ],
        "@platforms//os:windows": [
            #            "ODIN_USE_GSL",
        ],
        "//conditions:default": [
            "ODIN_USE_GSL",
        ],
    }),
    #    defines = [
    #        "CORE_VERSION_INFO=\"0.1.0\"",
    #    ],
    includes = ["pydin/include"],
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":core_libs",
    ],
)

py_binary(
    name = "stub_generator",
    srcs = ["tools/stub_generator.py"],
    deps = [
        requirement("pybind11-stubgen"),
        requirement("black"),
    ],
)

pybind_stubgen(
    name = "core_stubs",
    src = ":core",
    code_formatter = ":black",
    ignore_invalid = [
        "signature",
        "defaultarg",
    ],
    log_level = "DEBUG",
    module_name = "core",
    no_setup_py = True,
    root_module_suffix = "",
    tool = ":stub_generator",
)

## pydin library ####################################################################################
alias(
    name = "black",
    actual = entry_point("black", "black"),
)

genrule(
    name = "__init__",
    outs = ["__init__.py"],
    cmd = """
    OUT_FILE=$$(mktemp)
    echo "from .core import *" > $${OUT_FILE}
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
        ":core",
        ":core_stubs",
    ],
    prefix = "pydin",
    visibility = ["//visibility:private"],
)

pkg_files(
    name = "pydin_tests",
    srcs = ["//tests:test_files"],
    prefix = "tests",
    visibility = ["//visibility:private"],
)

pkg_files(
    name = "pydin_examples",
    srcs = ["//examples:example_files"],
    prefix = "examples",
    visibility = ["//visibility:private"],
)

pkg_files(
    name = "environment_files",
    srcs = [
        "//:environment.yml",
        "//:requirements.txt",
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

py_library(
    name = "pydin",
    data = [
        ":__init__",
        ":core",
        ":core_stubs",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [requirement("numpy")],
)

## pybind11 pydin stubs #############################################################################
py_binary(
    name = "foo_test",
    srcs = ["foo_test.py"],
    deps = [
        ":pydin",
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("matplotlib"),
    ],
)
