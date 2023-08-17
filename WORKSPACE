# ---------------------------------------------------------
# Workspace name
# ---------------------------------------------------------
workspace(name = "pydin")

local_repository(
    name = "pybind11_bazel",
    path = "external/pybind11_bazel",
)

local_repository(
    name = "odin",
    path = "../odin",
)

load("@odin//:repositories.bzl", "odin_dependencies")

odin_dependencies()

#local_repository(
#    name = "pydin",
#    path = ".",
#)

load("@pydin//:repositories.bzl", "pydin_dependencies")

pydin_dependencies()

# --------------------------------------------------------- LLVM
#load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")

#bazel_toolchain_dependencies()

#load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm", "llvm_toolchain")

#llvm_toolchain(
#    name = "llvm_toolchain",
#    llvm_version = "16.0.0",
#)

#llvm(
#    name = "llvm",
#    llvm_version = "16.0.0",
#)

#load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

#llvm_register_toolchains()
# --------------------------------------------------------- LLVM

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_pkg//pkg:deps.bzl", "rules_pkg_dependencies")

rules_foreign_cc_dependencies()

rules_pkg_dependencies()

#register_execution_platforms(
#    ":x64_windows-clang-cl",
#)
#
#register_toolchains(
#    "@local_config_cc//:cc-toolchain-x64_windows-clang-cl",
#)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_11",
    python_version = "3.11",
)

load("@rules_python//python:pip.bzl", "pip_install", "pip_parse")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
load("@python3_11//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_install", "pip_parse")

python_configure(
    name = "local_config_python",
    python_interpreter_target = interpreter,
)

load("@rules_python//python:pip.bzl", "compile_pip_requirements", "pip_parse")

#compile_pip_requirements(
#    name = "req",
#    requirements_in = "@pydin//:requirements.in",
#    requirements_txt = "requirementss",
#)

pip_parse(
    name = "pip",

    # Here, we use the interpreter constant that resolves to the host interpreter from the default Python toolchain.
    python_interpreter_target = interpreter,

    # Uses the default repository name "pip"
    requirements_lock = "//:requirements_lock.txt",
    #    requirements_lock = "//:requirements_lock.txt",
    #    requirements_lock = ":req",
    requirements_windows = "//:requirements_windows.txt",
)

load("@pip//:requirements.bzl", "install_deps")

# Initialize repositories for all packages in requirements.txt.
install_deps()
