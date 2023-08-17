# deps.bzl
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def pydin_dependencies():
    #    maybe(
    #        http_archive,
    #        name = "rules_python",
    #        sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
    #        strip_prefix = "rules_python-0.23.1",
    #        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.23.1.tar.gz",
    #    )

    maybe(
        git_repository,
        name = "rules_python",
        remote = "https://github.com/bazelbuild/rules_python.git",
        commit = "49d2b7aadb084ac7cae48583c38af6da2ce41a02",
    )

    maybe(
        git_repository,
        name = "github_bodgergely_spdlog_python",
        remote = "https://github.com/bodgergely/spdlog-python.git",
        commit = "3f32b0f2d7f249c3a9c9e67f44eb45ca9a144f8b",
        build_file = "@pydin//:external/spdlog_python.BUILD",
    )
    #
    #    maybe(
    #        http_archive,
    #        name = "github_llvm_llvm_project",
    #        urls = ["https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.6.tar.gz"],
    #        strip_prefix = "llvm-project-llvmorg-16.0.6",
    #        sha256 = "56b2f75fdaa95ad5e477a246d3f0d164964ab066b4619a01836ef08e475ec9d5",
    #        build_file = "@pydin//:external/llvm_project.BUILD",
    #    )

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
        build_file = "@pydin//:external/pybind11.BUILD",
        sha256 = "115bc49b69133dd8a7a7f824b669826ff6a35bc70a28ce2cf987d57c50a6543a",
        strip_prefix = "pybind11-2.10.4",
        urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.zip"],
    )

    maybe(
        http_file,
        name = "eros_50k_ply",
        urls = ["https://3d-asteroids.space/data/asteroids/models/e/433_Eros_50k.ply"],
        #        sha256 = "<sha256-value>",  # You'll need to calculate the sha256 value for the file
        downloaded_file_path = "eros_50k.ply",
    )

    # LLVM
    #    BAZEL_TOOLCHAIN_TAG = "0.8.2"
    #    BAZEL_TOOLCHAIN_SHA = "0fc3a2b0c9c929920f4bed8f2b446a8274cad41f5ee823fd3faa0d7641f20db0"
    #    maybe(
    #        git_repository,
    #        name = "com_grail_bazel_toolchain",
    #        remote = "https://github.com/grailbio/bazel-toolchain.git",
    #        commit = "f94335f1f5434256b1793dafbb7dd07773b0e76e",
    #    )
    maybe(
        http_archive,
        name = "llvm",
        urls = ["https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz"],
        build_file = "@pydin//:external/llvm.BUILD",
        sha256 = "2b8a69798e8dddeb57a186ecac217a35ea45607cb2b3cf30014431cff4340ad1",
        strip_prefix = "clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04",
    )

#    maybe(
#        http_archive,
#        name = "com_grail_bazel_toolchain",
#        sha256 = BAZEL_TOOLCHAIN_SHA,
#        strip_prefix = "bazel-toolchain-{tag}".format(tag = BAZEL_TOOLCHAIN_TAG),
#        canonical_id = BAZEL_TOOLCHAIN_TAG,
#        url = "https://github.com/grailbio/bazel-toolchain/archive/refs/tags/{tag}.tar.gz".format(tag = BAZEL_TOOLCHAIN_TAG),
#    )
