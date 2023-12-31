load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "filter_directory", "pkg_files", "pkg_mklink", "strip_prefix")
load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)

#exports_files(["MODULE.bazel"])

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

## PYBIND EXTENSION: CORE ###############################################
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "use_esa_polyhedral_gravity",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "esa_polyhedral_gravity",
    flag_values = {"use_esa_polyhedral_gravity": "True"},
    #    values = {"define": "USE_ESA_POLYHEDRAL_GRAVITY"},
)

bool_flag(
    name = "use_gsl_ellipsoidal_gravity",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "gsl_ellipsoidal_gravity",
    flag_values = {"use_gsl_ellipsoidal_gravity": "True"},
    #    values = {"define": "USE_GSL_ELLIPSOIDAL_GRAVITY"},
)

config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
)

CORE_LINKOPTS = select({
    ":msvc_compiler": [
    ],
    "@platforms//os:osx": [
    ],
    "//conditions:default": [
        "-fopenmp",  # Add the -fopenmp flag
    ],
})

CORE_COPTS = select({
    ":msvc_compiler": [
    ],
    "@platforms//os:osx": [
        "-Xpreprocessor",
        "-fopenmp",  # Add the -fopenmp flag
    ],
    "//conditions:default": [
        "-fopenmp",  # Add the -fopenmp flag
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

#ifdef USE_ESA_POLYHEDRAL_GRAVITY

pybind_library(
    name = "core_libs",
    hdrs = [":core_hdrs"],
    copts = CORE_COPTS,
    defines = [
        "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG",
        "PYBIND11_DETAILED_ERROR_MESSAGES",
    ] + select({
        ":esa_polyhedral_gravity": ["USE_ESA_POLYHEDRAL_GRAVITY"],
        "//conditions:default": [],
    }) + select({
        ":gsl_ellipsoidal_gravity": ["USE_GSL_ELLIPSOIDAL_GRAVITY"],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        #        "@com_github_google_glog//:glog",
        #        "@com_github_oneapi_onetbb//:tbb",
        #        "@com_github_uscilab_cereal//:cereal",
        "@odin",
        "@github_llvm_llvm_project//:llvm",
        #        "@com_github_autodiff_autodiff//:autodiff",
    ] + select({
        ":gsl_ellipsoidal_gravity": [
            "@org_gnu_gsl//:gsl",
        ],
        "//conditions:default": [],
    }) + select({
        ":esa_polyhedral_gravity": [
            "@com_github_esa_polyhedral_gravity_model//:polyhedral_gravity",
        ],
        "//conditions:default": [],
    }),
)

pybind_extension(
    name = "core",
    srcs = ["core.cpp"],
    copts = CORE_COPTS,
    defines = select({
        "@platforms//os:osx": [
            #            "ODIN_USE_GSL",
        ],
        "@platforms//os:windows": [
            #            "ODIN_USE_GSL",
        ],
        "//conditions:default": [
            "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG",
            "ODIN_USE_GSL",
        ],
    }),
    #    includes = ["pydin/include"],
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":core_libs",
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

filegroup(
    name = "py.files",
    srcs = glob([
        "**/*.py",
        "__init__.py",
    ]),
)

py_library(
    name = "pydin",
    srcs = [
        ":py.files",
    ],
    data = [
        ":__init__",
        ":core",
        ":core_stubs",
        #        "@github_bodgergely_spdlog_python//:spdlog",
    ],
    imports = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
    ],
)

## pybind11 pydin stubs #############################################################################
py_binary(
    name = "foo_test",
    srcs = ["foo_test.py"],
    data = [
        "//tests:data/Eros.face",
        "//tests:data/Eros.node",
        "@eros_50k_ply//file",
    ],
    imports = [
        ".",
        "pydin",
    ],
    legacy_create_init = False,
    deps = [
        "//pydin",
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("pillow"),
        requirement("pyvista"),

        # mayavi
        requirement("mayavi"),
        requirement("PySide2"),
        requirement("matplotlib"),
        #        entry_point("setuptools", "easy_install"),
        # @pip_setuptools
    ],
)
