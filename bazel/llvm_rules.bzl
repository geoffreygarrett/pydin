# llvm_rules.bzl

def generate_llvm_rules():
    hdrs = native.glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inc",
        "include/**/*.def",
    ])

    native.filegroup(
        name = "llvm_includes",
        srcs = hdrs,
        visibility = ["//visibility:public"],
    )

    llvm_libs = native.glob(["lib/*.a"])
    lib_targets = []
    for lib in llvm_libs:
        lib_target = lib.replace("/", "_").replace(".", "_")
        lib_targets.append(":" + lib_target)
        native.cc_import(
            name = lib_target,
            static_library = lib,
            hdrs = [":llvm_includes"],
            visibility = ["//visibility:public"],
        )

    native.cc_library(
        name = "llvm_headers",
        hdrs = [":llvm_includes"],
        includes = ["include"],
        visibility = ["//visibility:public"],
    )

    def filter_libs(lib_list):
        return [lib for lib in lib_list if "libbolt_rt_hugify.a" not in lib and "libbolt_rt_instr_osx.a" not in lib and "libbolt_rt_instr.a" not in lib]

    filtered_libs_x86_64 = filter_libs(lib_targets)

    native.cc_library(
        name = "llvm",
        deps = select({
            ":x86_64": filtered_libs_x86_64,
            ":arm": lib_targets,  # include all for ARM, modify as needed
            "//conditions:default": [],
        }),
        hdrs = [":llvm_includes"],
        visibility = ["//visibility:public"],
    )
