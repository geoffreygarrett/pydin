load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "-v",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
    requirements_windows = "requirements_windows.txt",
    requirements_darwin = "requirements_darwin.txt",
    requirements_linux = "requirements_linux.txt",
    visibility = ["//visibility:public"],
)