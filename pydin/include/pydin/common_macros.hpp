#define DEF_IO_BINARY_METHODS(cls, default_portable)                                              \
    def_static(                                                                                   \
            "from_binary",                                                                        \
            [](const std::string &binary_str, bool portable) {                                    \
                return from_binary<std::shared_ptr<cls>>(binary_str, portable);                   \
            },                                                                                    \
            "binary_str"_a,                                                                       \
            "portable"_a = default_portable)                                                      \
            .def_static(                                                                          \
                    "load_binary",                                                                \
                    [](const std::string &filename, bool portable) {                              \
                        return load_binary<std::shared_ptr<cls>>(filename, portable);             \
                    },                                                                            \
                    "filename"_a,                                                                 \
                    "portable"_a = default_portable)                                              \
            .def(                                                                                 \
                    "to_binary",                                                                  \
                    [](std::shared_ptr<cls> &self, bool portable) {                               \
                        return to_binary(self, portable);                                         \
                    },                                                                            \
                    "portable"_a = default_portable)                                              \
            .def(                                                                                 \
                    "save_binary",                                                                \
                    [](std::shared_ptr<cls> &self, const std::string &filename, bool portable) {  \
                        save_binary(self, filename, portable);                                    \
                    },                                                                            \
                    "filename"_a,                                                                 \
                    "portable"_a = default_portable)


#define DEF_IO_JSON_METHODS(cls)                                                                  \
    def_static(                                                                                   \
            "from_json",                                                                          \
            [](const std::string &json_str) {                                                     \
                return from_json<std::shared_ptr<cls>>(json_str);                                 \
            },                                                                                    \
            "json_str"_a)                                                                         \
            .def_static(                                                                          \
                    "load_json",                                                                  \
                    [](const std::string &filename) {                                             \
                        return load_json<std::shared_ptr<cls>>(filename);                         \
                    },                                                                            \
                    "filename"_a)                                                                 \
            .def("to_json", [](std::shared_ptr<cls> &self) { return to_json(self); })             \
            .def(                                                                                 \
                    "save_json",                                                                  \
                    [](std::shared_ptr<cls> &self, const std::string &filename) {                 \
                        save_json(self, filename);                                                \
                    },                                                                            \
                    "filename"_a)

#define DEF_PY_BINARY_PICKLING(cls)                                                               \
    def(py::pickle(                                                                               \
            [](const std::shared_ptr<cls> &obj) {                                                 \
                std::string binary_str = to_binary(obj, true);                                    \
                return py::bytes(binary_str);                                                     \
            },                                                                                    \
            [](const py::bytes &state) {                                                          \
                std::string binary_str = state.cast<std::string>();                               \
                return from_binary<std::shared_ptr<cls>>(binary_str, true);                       \
            }))

#define DEF_REPR(cls)                                                                             \
    def("__repr__", [](const cls &obj) {                                                          \
        std::ostringstream os;                                                                    \
        os << obj;                                                                                \
        return os.str();                                                                          \
    })