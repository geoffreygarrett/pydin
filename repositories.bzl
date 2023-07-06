# deps.bzl
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def pydin_dependencies():
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
        strip_prefix = "rules_python-0.23.1",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.23.1.tar.gz",
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
        sha256 = "b72c5b44135b90d1ffaba51e08240be0b91707ac60bea08bb4d84b47316211bb",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "pybind11.BUILD",
        sha256 = "115bc49b69133dd8a7a7f824b669826ff6a35bc70a28ce2cf987d57c50a6543a",
        strip_prefix = "pybind11-2.10.4",
        urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
    )
