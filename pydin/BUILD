load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library", "pybind_stubgen")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("@rules_pkg//pkg:mappings.bzl", "filter_directory", "pkg_files", "pkg_mklink", "strip_prefix")
load(
    "@pip_build//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
)
load(
    "@pip_example//:requirements.bzl",
    example_requirement = "requirement",
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
    flag_values = {
        "use_gsl_ellipsoidal_gravity": "False",
    },
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
        #        "-lomp",
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
        "-fPIC",
        "-std=c++20",
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

#LLVM_DEPS = [
#    "@llvm//:llvm_headers",
#    "@llvm//:lib_libLLVMSupport_a",
#    "@llvm//:lib_libLLVMDemangle_a",
#    "@llvm//:lib_libLLVMBinaryFormat_a",
#    "@llvm//:lib_libLLVMCore_a",
#    "@llvm//:lib_libLLVMMC_a",
#    "@llvm//:lib_libLLVMMCDisassembler_a",
#    "@llvm//:lib_libLLVMBitReader_a",
#    "@llvm//:lib_libLLVMMCParser_a",
#    "@llvm//:lib_libLLVMAnalysis_a",
#    "@llvm//:lib_libLLVMObject_a",
#    "@llvm//:lib_libLLVMTransformUtils_a",
#    "@llvm//:lib_libLLVMInstCombine_a",
#    "@llvm//:lib_libLLVMScalarOpts_a",
#    "@llvm//:lib_libLLVMTarget_a",
#    "@llvm//:lib_libLLVMX86CodeGen_a",
#    "@llvm//:lib_libLLVMAsmPrinter_a",
#    "@llvm//:lib_libLLVMCodeGen_a",
#    "@llvm//:lib_libLLVMExecutionEngine_a",
#]

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
    includes = select({
        "//conditions:default": ["include"],
    }),
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        #        "@com_github_google_glog//:glog",
        #        "@com_github_oneapi_onetbb//:tbb",
        #        "@com_github_uscilab_cereal//:cereal",
        "//odin",
        "@libomp//:omp",
        "@brew_gsl//:gsl",
        #        "@github_llvm_llvm_project//:llvm",
        #        "@llvm",
        #        "@llvm_toolchain//:all-components-x86_64-linux",

        #        "@com_github_autodiff_autodiff//:autodiff",
    ] + select({
        ":gsl_ellipsoidal_gravity": [
            #            "@org_gnu_gsl//:gsl",
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

        #        "@llvm",
        #        "@llvm//:lib_libLLVMCore_a",
        #        "@llvm//:lib_libLLVMIRReader_a",
        #        "@llvm//:lib_libLLVMSupport_a",
        #        "@llvm//:llvm_headers",
        #        "@odin",
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
    srcs = ["//pydin:tools/stub_generator.py"],
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

exports_files(["py.typed"])

py_library(
    name = "pydin",
    srcs = [
        ":py.files",
    ],
    data = [
        ":__init__",
        ":core",
        #        ":core_stubs",
        #        "@github_bodgergely_spdlog_python//:spdlog",
    ],
    imports = [
        "..",
        #        "bor/pydin",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
    ],
)

py_binary(
    name = "dev",
    srcs = ["dev.py"],
    data = [
        "//pydin/tests:data/Eros.face",
        "//pydin/tests:data/Eros.node",
        "@eros_50k_ply//file",
    ],
    imports = [
        #        ".",
        "bor/pydin",
    ],
    legacy_create_init = False,
    visibility = ["//visibility:public"],
    deps = [
        "@@bor//pydin/pydin",
        example_requirement("numpy"),
        example_requirement("pandas"),
        example_requirement("numba"),
        example_requirement("pillow"),
        example_requirement("sympy"),
        #requirement("pyvista"),

        # mayavi
        example_requirement("mayavi"),
        example_requirement("trimesh"),
        example_requirement("pyqt6"),
        example_requirement("pyside6"),
        #        entry_point("traitsui", "qt4"),
        #        requirement("wxPython"),
        #        requirement("pyside2"),
        #        requirement("pyside2"),
        example_requirement("matplotlib"),
        #        entry_point("setuptools", "easy_install"),
        # @pip_setuptools
    ],
)
