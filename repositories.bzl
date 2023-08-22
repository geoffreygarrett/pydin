# deps.bzl
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

DEPS = [
    {
        "function": git_repository,
        "name": "rules_python",
        "remote": "https://github.com/bazelbuild/rules_python.git",
        "commit": "49d2b7aadb084ac7cae48583c38af6da2ce41a02",
    },
    {
        "function": git_repository,
        "name": "github_bodgergely_spdlog_python",
        "remote": "https://github.com/bodgergely/spdlog-python.git",
        "commit": "3f32b0f2d7f249c3a9c9e67f44eb45ca9a144f8b",
        "build_file": "//pydin:external/spdlog_python.BUILD",
    },
    {
        "function": http_archive,
        "name": "pybind11_bazel",
        "strip_prefix": "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
        "sha256": "b72c5b44135b90d1ffaba51e08240be0b91707ac60bea08bb4d84b47316211bb",
        "urls": ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
    },
    {
        "function": http_archive,
        "name": "pybind11",
        "build_file": "//pydin:external/pybind11.BUILD",
        "sha256": "115bc49b69133dd8a7a7f824b669826ff6a35bc70a28ce2cf987d57c50a6543a",
        "strip_prefix": "pybind11-2.10.4",
        "urls": ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
    },
    {
        "function": http_file,
        "name": "eros_50k_ply",
        "urls": ["https://3d-asteroids.space/data/asteroids/models/e/433_Eros_50k.ply"],
        "downloaded_file_path": "eros_50k.ply",
    },
    {
        "function": http_archive,
        "name": "llvm",
        "urls": ["https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz"],
        "build_file": "//pydin:external/llvm.BUILD",
        "sha256": "2b8a69798e8dddeb57a186ecac217a35ea45607cb2b3cf30014431cff4340ad1",
        "strip_prefix": "clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04",
    },
]

def pydin_dependencies():
    for dep in DEPS:
        maybe(
            dep["function"],
            name = dep["name"],
            **{k: v for k, v in dep.items() if k not in ["function", "name"]}
        )
