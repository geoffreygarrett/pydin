load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "-v",
    ],
    requirements_darwin = "requirements_darwin.txt",
    requirements_in = "requirements.in",
    requirements_linux = "requirements_linux.txt",
    requirements_txt = "requirements_lock.txt",
    requirements_windows = "requirements_windows.txt",
    visibility = ["//visibility:public"],
)
