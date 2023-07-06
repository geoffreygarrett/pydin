## extensions.bzl
load("//:repositories.bzl", "pydin_dependencies")
load("//:python_configure.bzl", "python_configure")

def _init(ctx):
    pydin_dependencies()

    for mod in ctx.modules:
        for configure in mod.tags.configure:
            python_configure(
                name = configure.name,
                python_version = configure.python_version,
                python_interpreter_target = configure.python_interpreter_target,
            )

def _python_configure():
    attrs = dict({
        "name": attr.string(default = "local_config_python"),
        "python_version": attr.string(default = ""),
        "python_interpreter_target": attr.label(),
    })
    return attrs

init = module_extension(
    tag_classes = {
        "configure": tag_class(attrs = _python_configure()),
    },
    implementation = _init,
)
