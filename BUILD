load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
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

## PYBIND EXTENSION: CORE ###############################################
filegroup(
    name = "pydin_ext_headers",
    srcs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
        "include/**/*.cuh",
    ]),
)

# "-fexperimental-library",
# "-DCC=$$(CC)",
# "-DCXX=$$(CXX)",
# "-DLDFLAGS=$$(LDFLAGS)",
# "-DCPPFLAGS=$$(CPPFLAGS)",

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

cc_library(
    name = "pydin_ext_lib",
    hdrs = [
        ":pydin_ext_headers",
    ],
    copts = CORE_COPTS,
    defines = [
        "PYBIND11_DETAILED_ERROR_MESSAGES",
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
    ],
)

pybind_extension(
    name = "core",
    srcs = ["core.cpp"],
    copts = CORE_COPTS,
    includes = ["pydin/include"],
    linkopts = CORE_LINKOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":pydin_ext_lib",
    ],
)

#
## pydin library ####################################################################################
genrule(
    name = "pydin_init",
    outs = ["__init__.py"],
    cmd = """
echo \"\"\"from .core import *\"\"\" > $(OUTS)
""",
)

genrule(
    name = "pydin_setup",
    outs = ["setup.py"],
    cmd = """
echo \"\"\"from setuptools import setup, find_packages

setup(
    name='pydin',
    version='0.1',
    packages=find_packages(),
    package_data={
        'pydin': ['*.so'],
    },
    install_requires=[
        'numpy',
        'pandas',
    ],
)\"\"\" > $(OUTS)
""",
)

filegroup(
    name = "pydin_files",
    srcs = [
        ":core.so",
        ":pydin_init",
        ":pydin_setup",
    ],
)

py_library(
    name = "pydin",
    srcs = glob(["__init__.py"]),
    data = [":pydin_files"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        #        requirement("pandas"),
    ],
)

#genrule(
#    name = "pydin-tar",
#    srcs = [
#        ":pydin_files",
#        ":pydin",
#    ],
#    outs = ["pydin.tar.gz"],
#    cmd = """
#        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#        rm -r $(locations :pydin_files)
#        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#        EXEC_DIR=$(@D)/pydin
#        mkdir -p $$OUT_DIR
#        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#        mv $$OUT_DIR/core/* $$EXEC_DIR
#        rm -r $$OUT_DIR
#        tar -czf $(@D)/pydin.tar.gz -C $(@D)/pydin .
#    """,
#    tools = [
#        ":pybind11-stubgen",
#    ],
#)

STUB_OUTPUT_COMMON = [
    "pydin/core.so",
    "pydin/__init__.py",
    "pydin/core/__init__.pyi",
    "pydin/core/mip/__init__.pyi",
    "pydin/core/logging/__init__.pyi",
]

genrule(
    name = "pydin-with-stubs",
    srcs = [
        ":pydin_files",
        ":pydin",
    ],
    outs = STUB_OUTPUT_COMMON,
    cmd = select({
        "//conditions:default": """
        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
        EXEC_DIR=$(@D)/pydin
        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
        mkdir -p $$OUT_DIR
        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
        mv $$OUT_DIR/core/* $$EXEC_DIR/core/
        rm -r $$OUT_DIR
    """,
        "@platforms//os:windows": """
        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
        EXEC_DIR=$(@D)/pydin
        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
        mkdir -p $$OUT_DIR
        ln -s $$EXEC_DIR/core.pyd $$EXEC_DIR/core.so
        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
        mv $$OUT_DIR/core/* $$EXEC_DIR
        rm -r $$OUT_DIR
    """,
    }),
    tools = [
        ":pybind11-stubgen",
    ],
)

#genrule(
#    name = "pydin-with-stubs-windows",
#    srcs = [
#        ":pydin_files",
#        ":pydin",
#    ],
#    outs = STUB_OUTPUT_WINDOWS,
#    cmd = """
#        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#        EXEC_DIR=$(@D)/pydin
#        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#        mkdir -p $$OUT_DIR
#        # rename before executing core
#        mv $$EXEC_DIR/core.so $$EXEC_DIR/core.pyd
#        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#        mv $$OUT_DIR/core/* $$EXEC_DIR/core/
#        rm -r $$OUT_DIR
#    """,
#    tools = [
#        ":pybind11-stubgen",
#    ],
#)

pkg_mklink(
    name = "windows-symlink",  # Arbitrary name for this rule
    link_name = "core.pyd",  # The name of the link
    target = "core.so",  # What the link points to
)

pkg_zip(
    name = "pydin-zip",
    srcs = [
        ":pydin-with-stubs",
        ":windows-symlink",
    ],
    out = "pydin.zip",
    strip_prefix = strip_prefix.from_pkg("external/pydin/pydin"),
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "pydin-tar",
    srcs = [
        ":pydin-with-stubs",
    ],
    out = "pydin.tar.gz",
    strip_prefix = strip_prefix.from_pkg("external/pydin/pydin"),
    visibility = ["//visibility:public"],
)

## pybind11 pydin stubs #############################################################################
alias(
    name = "pybind11-stubgen",
    actual = entry_point("pybind11-stubgen", "pybind11-stubgen"),
)

genrule(
    name = "pydin-stubs-tar",
    srcs = [
        "pydin_files",
    ],
    outs = ["pydin-stubs.tar"],
    cmd = """
        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
        OUT_DIR=$$(realpath $(@D)/stubs)
        EXEC_DIR=$(@D)
        mkdir -p $$OUT_DIR
        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) pydin -o $$OUT_DIR
        tar -cf $(@D)/pydin-stubs.tar -C $$OUT_DIR .
    """,
    tools = [
        ":pybind11-stubgen",
    ],
)

#    cmd = """
#        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#        tar -cf $(@D)/pydin.tar -C $(@D)/pydin .
#    """,
## pydin package ####################################################################################
genrule(
    name = "package_setup",
    outs = ["package_setup.py"],
    cmd = """
echo \"\"\"from setuptools import setup, find_packages, Command
from setuptools.command.install import install
import os
import tarfile

class CustomInstall(install):
    def run(self):
        tar = tarfile.open('pydin.tar')
        tar.extractall()
        tar.close()
        tar = tarfile.open('pydin-stubs.tar')
        tar.extractall()
        tar.close()
        install.run(self)

setup(
    name='pydin',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'install': CustomInstall,
    },
    package_data={
        'pydin': ['*.so'],
    },
    install_requires=[
        'numpy',
        'pandas',
    ],
)\"\"\" > $(OUTS)
""",
)

filegroup(
    name = "package_files",
    srcs = [
        ":package_setup",
        ":pydin-stubs-tar",
        ":pydin-tar",
    ],
)

pkg_tar(
    name = "pydin_pkg",
    srcs = [
        ":package_files",
    ],
    extension = "tar.gz",
    #    strip_prefix = "/",
)

pkg_zip(
    name = "pydin_zip",
    srcs = [
        ":package_files",
    ],
    visibility = ["//visibility:package"],
    #    strip_prefix = "/",
)

# all tests
#py_test(
#    name = "all_tests",
#    srcs = glob(["*_test.py"]),
#    visibility = ["//visibility:public"],
#    deps = [
#        "//:tests
#    ],
#)

py_binary(
    name = "foo_test",
    srcs = ["foo_test.py"],
    deps = [
        ":pydin",
        requirement("numpy"),
        requirement("pandas"),
        requirement("numba"),
        requirement("llvmlite"),
    ],
)
