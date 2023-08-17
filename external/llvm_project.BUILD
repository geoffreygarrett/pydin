load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "llvm_sources",
    #    srcs = glob(["**"]),
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
    # Private is default, but just to be explicit. Sources would only be
    # part of the cmake target, but not part of the cc_library target.
)

filegroup(
    name = "llvm_headers",
    srcs = glob(["llvm/include/**"]),
    visibility = ["//visibility:public"],
    # We make this public, so we have the option of packaging it
    # with our output.
)

cmake(
    name = "llvm",
    lib_source = ":llvm_sources",
    working_directory = "llvm",
    generate_args = [
        #        "-G Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    ],
    out_static_libs = [
        "libLLVMCore.a",
        "libLLVMSupport.a",
        "libLLVMMC.a",
        "libLLVMIRReader.a",
        "libLLVMBitReader.a",
        "libLLVMBitWriter.a",
        "libLLVMTransformUtils.a",
        "libLLVMAnalysis.a",
        "libLLVMTarget.a",
        "libLLVMCodeGen.a",
        "libLLVMExecutionEngine.a",
        "libLLVMRuntimeDyld.a",
        "libLLVMJITLink.a",
        "libLLVMMCJIT.a",
        "libLLVMX86CodeGen.a",
        "libLLVMX86Desc.a",
        "libLLVMX86Info.a",
    ],
    build_args = [
        "-j$$(nproc)",
    ],
    cache_entries = {
        "LLVM_INCLUDE_TOOLS": "ON",
        "LLVM_BUILD_TOOLS": "ON",
        "LLVM_INCLUDE_UTILS": "ON",
        "LLVM_BUILD_UTILS": "ON",
        "LLVM_INCLUDE_RUNTIMES": "ON",
        "LLVM_BUILD_RUNTIMES": "ON",
        "LLVM_BUILD_RUNTIME": "ON",
        "LLVM_BUILD_EXAMPLES": "OFF",
        "LLVM_INCLUDE_EXAMPLES": "OFF",
        "LLVM_BUILD_TESTS": "OFF",
        "LLVM_INCLUDE_TESTS": "ON",
        "LLVM_BUILD_BENCHMARKS": "OFF",
        "LLVM_INCLUDE_BENCHMARKS": "OFF",
        "LLVM_BUILD_DOCS": "OFF",
        "LLVM_INCLUDE_DOCS": "OFF",
        "LLVM_ENABLE_DOXYGEN": "OFF",
        "LLVM_ENABLE_SPHINX": "OFF",
        "LLVM_ENABLE_OCAMLDOC": "OFF",
        "LLVM_ENABLE_BINDINGS": "OFF",
        #        "CMAKE_INSTALL_PREFIX": "llvm_install",  # Installation directory for the built libraries
        "CMAKE_BUILD_TYPE": "Release",  # Build type
        "LLVM_ENABLE_ASSERTIONS": "ON",  # Enable assertions
        #        "LLVM_USE_LINKER": "lld",  # Use lld linker
    },
    visibility = ["//visibility:public"],
)
