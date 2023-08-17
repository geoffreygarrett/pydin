# llvm.BUILD
load("@pydin//:llvm_rules.bzl", "generate_llvm_rules")

constraint_setting(name = "cpu_arch")

constraint_value(
    name = "x86_64",
    constraint_setting = ":cpu_arch",
)

constraint_value(
    name = "arm",
    constraint_setting = ":cpu_arch",
)

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        ":x86_64",
    ],
)

platform(
    name = "linux_arm",
    constraint_values = [
        "@platforms//os:linux",
        ":arm",
    ],
)

config_setting(
    name = "config_x86_64",
    values = {"cpu": "x86_64"},
)

config_setting(
    name = "config_arm",
    values = {"cpu": "arm"},
)

generate_llvm_rules()
