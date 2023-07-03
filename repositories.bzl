# repositories.bzl
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def pybind11_dependency():
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
        #        sha256 = "e1fe52ad3468629772c50c67a93449f235aed650a0fe8e89a22fbff285f677a1",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
    )

    http_archive(
        name = "pybind11",
        build_file = "external/pybind11.BUILD",
        sha256 = "115bc49b69133dd8a7a7f824b669826ff6a35bc70a28ce2cf987d57c50a6543a",
        strip_prefix = "pybind11-2.10.4",
        urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
    )
