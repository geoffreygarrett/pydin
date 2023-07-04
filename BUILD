load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load(
    "@pip//:requirements.bzl",
    "data_requirement",
    "entry_point",
    "requirement",
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

cc_library(
    name = "pydin_ext_lib",
    hdrs = [
        ":pydin_ext_headers",
    ],
    copts = select({
        "@bazel_tools//src/conditions:darwin": [
            "-fexperimental-library",
            "-DCC=$$(CC)",
            "-DCXX=$$(CXX)",
            "-DLDFLAGS=$$(LDFLAGS)",
            "-DCPPFLAGS=$$(CPPFLAGS)",
        ],
        "//conditions:default": [],
    }),
    defines = [
        "PYBIND11_DETAILED_ERROR_MESSAGES",
        "ODIN_USE_GLOG",
        "GLOG_CUSTOM_PREFIX_SUPPORT",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_google_glog//:glog",
        "@com_github_oneapi_onetbb//:tbb",
        "@com_github_uscilab_cereal//:cereal",
        "@odin",
    ],
)

pybind_extension(
    name = "core",
    srcs = ["core.cpp"],
    includes = ["pydin/include"],
    linkopts = [
        "-fopenmp",
        # "-fvisibility=default",  # https://groups.google.com/g/cerealcpp/c/qmpit5GEcZU
    ],
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
#    outs = ["pydin.tar"],
#    cmd = """
#        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#        rm -r $(locations :pydin_files)
#        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#        EXEC_DIR=$(@D)/pydin
#        mkdir -p $$OUT_DIR
#        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#        mv $$OUT_DIR/core/* $$EXEC_DIR
#        rm -r $$OUT_DIR
#        tar -cf $(@D)/pydin.tar -C $(@D)/pydin .
#    """,
#    tools = [
#        ":pybind11-stubgen",
#    ],
#)

#genrule(
#    name = "pydin-tar",
#    srcs = [
#        ":pydin_files",
#        ":pydin",
#    ],
#    outs = [
#        "pydin.tar.gz",
#        "pydin.zip",
#    ],
#    cmd = """
#        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#        rm -r $(locations :pydin_files)
#        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#        EXEC_DIR=$(@D)/pydin
#        mkdir -p $$OUT_DIR
#        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#        mv $$OUT_DIR/core/* $$EXEC_DIR
#        rm -r $$OUT_DIR
#        if [ -z $${OS+x} ]; then
#            OS=$$(uname -s)
#        fi
#        if [ "$$OS" == "Windows_NT" ] || [ "$$OS" == "MINGW64_NT-10.0" ]; then
#            # On Windows, use zip instead of tar
#            cd $(@D)/pydin
#            zip -r $(@D)/pydin.zip .
#        else
#            tar -czf $(@D)/pydin.tar.gz -C $(@D)/pydin .
#        fi
#    """,
#    tools = [
#        ":pybind11-stubgen",
#    ],
#)

#genrule(
#    name = "pydin-tar",
#    srcs = [
#        ":pydin_files",
#        ":pydin",
#    ],
#    outs = [
#        "pydin.tar.gz",
#        "pydin.zip",
#    ],
#    cmd = select({
#        "@bazel_tools//src/conditions:windows": """
#            mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#            rm -r $(locations :pydin_files)
#            OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#            EXEC_DIR=$(@D)/pydin
#            mkdir -p $$OUT_DIR
#            PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#            mv $$OUT_DIR/core/* $$EXEC_DIR
#            rm -r $$OUT_DIR
#            cd $(@D)/pydin
#            zip -r $(@D)/pydin.zip .
#        """,
#        "//conditions:default": """
#            mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
#            rm -r $(locations :pydin_files)
#            OUT_DIR=$$(realpath $(@D)/pydin-stubs)
#            EXEC_DIR=$(@D)/pydin
#            mkdir -p $$OUT_DIR
#            PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
#            mv $$OUT_DIR/core/* $$EXEC_DIR
#            rm -r $$OUT_DIR
#            tar -czf $(@D)/pydin.tar.gz -C $(@D)/pydin .
#        """,
#    }),
#    tools = [
#        ":pybind11-stubgen",
#    ],
#)
genrule(
    name = "pydin-tar",
    srcs = [
        ":pydin_files",
        ":pydin",
    ],
    outs = ["pydin.tar.gz"],
    cmd = """
        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
        rm -r $(locations :pydin_files)
        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
        EXEC_DIR=$(@D)/pydin
        mkdir -p $$OUT_DIR
        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
        mv $$OUT_DIR/core/* $$EXEC_DIR
        rm -r $$OUT_DIR
        tar -czf $(@D)/pydin.tar.gz -C $(@D)/pydin .
    """,
    tools = [
        ":pybind11-stubgen",
    ],
)

genrule(
    name = "pydin-zip",
    srcs = [
        ":pydin_files",
        ":pydin",
    ],
    outs = ["pydin.zip"],
    cmd = """
        mkdir -p $(@D)/pydin && cp $(locations :pydin_files) $(@D)/pydin
        rm -r $(locations :pydin_files)
        OUT_DIR=$$(realpath $(@D)/pydin-stubs)
        EXEC_DIR=$(@D)/pydin
        mkdir -p $$OUT_DIR
        PYTHONPATH=$$EXEC_DIR $(execpath :pybind11-stubgen) core -o $$OUT_DIR --root-module-suffix ''
        mv $$OUT_DIR/core/* $$EXEC_DIR
        rm -r $$OUT_DIR
        cd $(@D)/pydin
        zip -r $(@D)/pydin.zip .
    """,
    tools = [
        ":pybind11-stubgen",
    ],
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
    #    strip_prefix = "/",
)

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
