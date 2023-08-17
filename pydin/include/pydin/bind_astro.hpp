#ifndef PYDIN_INCLUDE_ASTRODYNAMICS_HPP
#define PYDIN_INCLUDE_ASTRODYNAMICS_HPP


#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <string>

#include <odin/domain/astrodynamics.hpp>

using namespace pybind11::literals;
namespace py = pybind11;


#define ARG_ANGLE py::arg("angle")
#define ARG_GM py::arg("mu")

#include <functional>

// Define the type of function that rot_wrapper will return
template<typename Scalar>
using rotation_function_type = std::function<Eigen::Matrix<Scalar, 3, 3>(Scalar)>;

// Wrapper function
template<int I, typename Scalar>
rotation_function_type<Scalar> rot_wrapper() {
    return [](float angle) { return rot<I, Scalar>(angle); };
}

template<typename Scalar>
class RotFunction {
public:
    rotation_function_type<Scalar> operator[](int i) const {
        if (i == 0) return rot_wrapper<0, Scalar>();
        else if (i == 1)
            return rot_wrapper<1, Scalar>();
        else if (i == 2)
            return rot_wrapper<2, Scalar>();
        else
            throw std::invalid_argument("Invalid index");
    }

    rotation_function_type<Scalar> operator[](const std::string &s) const {
        if (s == "x" || s == "X") return rot_wrapper<0, Scalar>();
        else if (s == "y" || s == "Y")
            return rot_wrapper<1, Scalar>();
        else if (s == "z" || s == "Z")
            return rot_wrapper<2, Scalar>();
        else
            throw std::invalid_argument("Invalid index");
    }

    rotation_function_type<Scalar> x() const { return rot_wrapper<0, Scalar>(); }
    rotation_function_type<Scalar> y() const { return rot_wrapper<1, Scalar>(); }
    rotation_function_type<Scalar> z() const { return rot_wrapper<2, Scalar>(); }
};


template<typename Float>
void bind_astrodynamics(py::module &m, const std::string &suffix = "") {
// define macro for suffixing function names
#define TYPE_SUFFIX(name) (name + suffix).c_str()

    // KEPLERIAN TOOLS

    // Rotation matrices
    m.def(TYPE_SUFFIX("rot1"), &rot<0, Float>, ARG_ANGLE);
    m.def(TYPE_SUFFIX("rot2"), &rot<1, Float>, ARG_ANGLE);
    m.def(TYPE_SUFFIX("rot3"), &rot<2, Float>, ARG_ANGLE);

    using rotation_function_type = RotFunction<Float>;
    py::class_<rotation_function_type>(m, "RotFunction")
            .def("__getitem__", [](const rotation_function_type &r, int i) { return r[i]; })
            .def("__getitem__",
                 [](const rotation_function_type &r, const std::string &s) { return r[s]; })
            .def_property_readonly("x", &rotation_function_type::x)
            .def_property_readonly("y", &rotation_function_type::y)
            .def_property_readonly("z", &rotation_function_type::z);

    m.attr("rot") = rotation_function_type();


    m.def(TYPE_SUFFIX("rv2coe"),
          py::overload_cast<Float, const Vector3<Float> &, const Vector3<Float> &, Float>(
                  &rv2coe<Float>),
          ARG_GM,
          py::arg("r"),
          py::arg("v"),
          py::arg("tol") = static_cast<Float>(1e-8));


    m.def(TYPE_SUFFIX("rv2coe"),
          py::overload_cast<Float,
                            const std::vector<Vector3<Float>> &,
                            const std::vector<Vector3<Float>> &,
                            Float>(&rv2coe<Float>),
          ARG_GM,
          py::arg("rr"),
          py::arg("vv"),
          py::arg("tol") = static_cast<Float>(1e-8));

    m.def(TYPE_SUFFIX("rv2coe"),
          &coe2pqr<Float>,
          ARG_GM,
          py::arg("p"),
          py::arg("e"),
          py::arg("nu"));

    m.def(TYPE_SUFFIX("coe2rv"),
          py::overload_cast<Float, Float, Float, Float, Float, Float, Float>(&coe2rv<Float>),
          ARG_GM,
          py::arg("p"),
          py::arg("e"),
          py::arg("i"),
          py::arg("raan"),
          py::arg("argp"),
          py::arg("nu"));

    m.def(TYPE_SUFFIX("coe2rv"),
          py::overload_cast<const Float,
                            const std::vector<Float> &,
                            const std::vector<Float> &,
                            const std::vector<Float> &,
                            const std::vector<Float> &,
                            const std::vector<Float> &,
                            const std::vector<Float> &>(&coe2rv<Float>),
          ARG_GM,
          py::arg("p"),
          py::arg("e"),
          py::arg("i"),
          py::arg("raan"),
          py::arg("argp"),
          py::arg("nu"));

    m.def(TYPE_SUFFIX("coe2rv"),
          py::overload_cast<Float, Float, Float, Float, Float, Float, const std::vector<Float> &>(
                  &coe2rv<Float>),
          ARG_GM,
          py::arg("p"),
          py::arg("e"),
          py::arg("i"),
          py::arg("raan"),
          py::arg("argp"),
          py::arg("nu"));

    // Angle routines
    m.def(TYPE_SUFFIX("anomaly_mean_to_eccentric"),
          &anomaly_mean_to_eccentric<Float>,
          py::arg("M"),
          py::arg("e"),
          py::arg("max_iter")  = 1000,
          py::arg("tolerance") = 1e-6);

    m.def(TYPE_SUFFIX("anomaly_eccentric_to_mean"),
          &anomaly_eccentric_to_mean<Float>,
          py::arg("E"),
          py::arg("e"));

    m.def(TYPE_SUFFIX("anomaly_true_to_eccentric"),
          &anomaly_true_to_eccentric<Float>,
          py::arg("f"),
          py::arg("e"));

    m.def(TYPE_SUFFIX("anomaly_eccentric_to_true"),
          &anomaly_eccentric_to_true<Float>,
          "E"_a,
          "e"_a);

    m.def(TYPE_SUFFIX("anomaly_mean_to_true"),
          py::overload_cast<Float, Float, int, Float>(&anomaly_mean_to_true<Float>),
          "M"_a,
          "e"_a,
          "max_iter"_a = 1000,
          "tol"_a      = static_cast<Float>(1e-6));

    m.def(TYPE_SUFFIX("anomaly_mean_to_true"),
          py::overload_cast<std::vector<Float>, std::vector<Float>, int, Float>(
                  &anomaly_mean_to_true<Float>),
          "MM"_a,
          "ee"_a,
          "max_iter"_a = 1000,
          "tol"_a      = static_cast<Float>(1e-6));

    m.def(TYPE_SUFFIX("anomaly_true_to_mean"), &anomaly_true_to_mean<Float>, "f"_a, "e"_a);

    m.def(TYPE_SUFFIX("sample_true_from_eccentric_anomaly"),
          &sample_true_from_eccentric_anomaly<Float>,
          "e"_a,
          "n_points"_a,
          "Sample a Keplerian orbit in the eccentric anomaly domain and return the true "
          "anomalies.");

    m.def(TYPE_SUFFIX("sample_true_from_mean_anomaly"),
          &sample_true_from_mean_anomaly<Float>,
          "e"_a,
          "n_points"_a,
          "Sample a Keplerian orbit in the mean anomaly domain and return the true anomalies.");
}


#endif//PYDIN_INCLUDE_ASTRODYNAMICS_HPP