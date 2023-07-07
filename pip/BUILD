load("@rules_python//python:pip.bzl", "compile_pip_requirements")

exports_files(
    [
        "environment.yml",
        "requirements.in",
        "requirements_windows.txt",
        "requirements_lock.txt",
        #    "requirements.txt",
    ],
    visibility = ["//visibility:public"],
)

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
    requirements_in = ":requirements.in",
    requirements_txt = ":requirements_lock.txt",
    requirements_windows = ":requirements_windows.txt",
    visibility = ["//visibility:public"],
)
