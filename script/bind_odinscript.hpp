#ifndef ODINSCRIPT_H
#define ODINSCRIPT_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

class CompiledFunction {
public:
    std::string func_name{};
    py::list    args{};
    std::string return_type{};
    py::list    body{};
    py::object  func{};

    // constructor
    CompiledFunction(const std::string &name,
                     const py::list    &args,
                     const std::string &ret_type,
                     const py::list    &body,
                     const py::object  &func)
        : func_name(name), args(args), return_type(ret_type), body(body), func(func) {}

    // function to make the object callable
    py::object operator()(py::args args, py::kwargs kwargs) { return func(*args, **kwargs); }

    // function to get the annotations
    py::dict __annotations__() {
        py::dict annotations;
        for (auto item: args) {
            py::tuple arg                = item.cast<py::tuple>();
            annotations[py::str(arg[0])] = py::str(arg[1]);
        }
        annotations["return"] = return_type;
        return annotations;
    }

    // function to get the variable names
    py::tuple __code__() {
        py::list varnames;
        for (auto item: args) {
            py::tuple arg = item.cast<py::tuple>();
            varnames.append(py::str(arg[0]));
        }
        return varnames;
    }
};

py::dict parse_function(py::object func) {
    py::dict result;

    py::object inspect    = py::module_::import("inspect");
    py::object ast_module = py::module_::import("ast");

    // get the source code
    std::string source = py::str(inspect.attr("getsource")(func));

    // parse the source code into an AST
    py::object parsed = ast_module.attr("parse")(source);

    // get the function's signature
    py::object signature = inspect.attr("signature")(func);

    result["ast"]       = parsed;
    result["signature"] = signature;

    // start extracting information from the AST
    py::object ast_body = parsed.attr("body").attr("__getitem__")(
            0);// get the first (and only) element in the body
    std::string func_name = py::str(ast_body.attr("name"));
    result["func_name"]   = func_name;

    py::list   args_list;
    py::object args = ast_body.attr("args").attr("args");
    for (int i = 0; i < py::len(args); ++i) {
        args_list.append(py::make_tuple(
                py::str(args.attr("__getitem__")(i).attr("arg")),
                py::str(args.attr("__getitem__")(i).attr("annotation").attr("id"))));
    }
    result["args"] = args_list;

    std::string return_type = py::str(ast_body.attr("returns").attr("id"));
    result["return_type"]   = return_type;

    py::list   body_list;
    py::object body = ast_body.attr("body");
    for (int i = 0; i < py::len(body); ++i) {
        body_list.append(py::str(ast_module.attr("dump")(body.attr("__getitem__")(i))));
    }
    result["body"] = body_list;

    return result;
}

CompiledFunction jit(py::object func) {
    py::dict func_info = parse_function(func);

    std::string func_name   = py::str(func_info["func_name"]);
    py::list    args        = func_info["args"].cast<py::list>();
    std::string return_type = py::str(func_info["return_type"]);
    py::list    body        = func_info["body"].cast<py::list>();

    return CompiledFunction(func_name, args, return_type, body, func);
}

template<typename Float = double>
void bind_odinscript(py::module &m) {
    m.def("jit", &jit);
    py::class_<CompiledFunction>(m, "CompiledFunction")
            .def(py::init<const std::string &,
                          const py::list &,
                          const std::string &,
                          const py::list &,
                          const py::object &>())
            .def_readonly("func_name", &CompiledFunction::func_name)
            .def_readonly("args", &CompiledFunction::args)
            .def_readonly("return_type", &CompiledFunction::return_type)
            .def_readonly("body", &CompiledFunction::body)
            .def("__call__", &CompiledFunction::operator())
            .def_property_readonly("__annotations__", &CompiledFunction::__annotations__)
            .def_property_readonly("__code__", &CompiledFunction::__code__);
    // .def("other_functions", &CompiledFunction::other_functions);
}

#endif//ODINSCRIPT_H
