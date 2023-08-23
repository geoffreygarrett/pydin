load("@rules_python//python:pip.bzl", "pip_install", "pip_parse")

def pip_dependency_set(name, interpreter, requirements_path):
    pip_parse(
        name = name,
        python_interpreter_target = interpreter,
        requirements_darwin = requirements_path + ":requirements_darwin.txt",
        requirements_linux = requirements_path + ":requirements_linux.txt",
        requirements_lock = requirements_path + ":requirements_lock.txt",
        requirements_windows = requirements_path + ":requirements_windows.txt",
        visibility = ["//visibility:public"],
    )
