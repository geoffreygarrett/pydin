#ifndef PYDIN_BIND_SHAPE_H
#define PYDIN_BIND_SHAPE_H

#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include <odin/eigen.h>


//#include <odin/models/shape/shape_base.h>
//#include <odin/models/shape/shape_concepts.h>
#include <odin/models/shape/ellipsoid_impl.h>
#include <odin/models/shape/sphere_impl.h>


using namespace pybind11::literals;

#include <spdlog/spdlog.h>

//template<typename T>
//class pydin_array {
//public:
//    pydin_array() : host_data_(nullptr), device_data_(nullptr) {}
//
//    explicit pydin_array(const T &host_data)
//        : host_data_(new T(host_data)),
//          device_data_(new thrust::device_vector<T>(to_device(host_data))) {
//        if (!host_data_ || !device_data_) {
//            spdlog::error("Memory allocation failed in pydin_array constructor.");
//        } else {
//            spdlog::debug("pydin_array object successfully created.");
//        }
//    }
//
//    pydin_array(const pydin_array<T> &other)// Copy constructor
//        : host_data_(new T(*other.host_data_)),
//          device_data_(new thrust::device_vector<T>(*other.device_data_)) {
//        if (!host_data_ || !device_data_) {
//            spdlog::error("Memory allocation failed in pydin_array copy constructor.");
//        } else {
//            spdlog::debug("pydin_array object successfully copied.");
//        }
//    }
//
//    pydin_array(pydin_array<T> &&other) noexcept// Move constructor
//        : host_data_(other.host_data_), device_data_(other.device_data_) {
//        other.host_data_   = nullptr;
//        other.device_data_ = nullptr;
//        spdlog::debug("pydin_array object successfully moved.");
//    }
//
//    pydin_array &operator=(const pydin_array<T> &other) {// Copy assignment operator
//        if (this != &other) {
//            delete host_data_;
//            delete device_data_;
//            host_data_   = new T(*other.host_data_);
//            device_data_ = new thrust::device_vector<T>(*other.device_data_);
//            if (!host_data_ || !device_data_) {
//                spdlog::error("Memory allocation failed in pydin_array copy assignment operator.");
//            } else {
//                spdlog::debug("pydin_array object successfully assigned.");
//            }
//        }
//        return *this;
//    }
//
//    pydin_array &operator=(pydin_array<T> &&other) noexcept {// Move assignment operator
//        if (this != &other) {
//            delete host_data_;
//            delete device_data_;
//            host_data_         = other.host_data_;
//            device_data_       = other.device_data_;
//            other.host_data_   = nullptr;
//            other.device_data_ = nullptr;
//            spdlog::debug("pydin_array object successfully move-assigned.");
//        }
//        return *this;
//    }
//
//    ~pydin_array() {// Destructor
//        delete host_data_;
//        delete device_data_;
//        spdlog::debug("pydin_array object successfully destroyed.");
//    }
//
//    const T &operator[](std::size_t idx) const {
//        if (idx < size()) {
//            return (*host_data_)[idx];
//        } else {
//            spdlog::error("Index out of bounds in pydin_array indexing operator.");
//            // handle error...
//            // Here you could throw an exception or return a sentinel value.
//            // For instance, throwing an exception would look like this:
//            throw std::out_of_range("Index out of bounds in pydin_array indexing operator.");
//        }
//    }
//
//    T &operator[](std::size_t idx) {// Indexing operator
//        if (idx < size()) {
//            return (*host_data_)[idx];
//        } else {
//            spdlog::error("Index out of bounds in pydin_array indexing operator.");
//            // handle error...
//
//            // Here you could throw an exception or return a sentinel value.
//
//        }
//    }

//
//    [[nodiscard]] std::size_t size() const { return host_data_->size(); }// Size function
//
//    T to_host() const {
//        // Implementation of conversion from device to host
//        spdlog::debug("Converted pydin_array object from device to host.");
//    }
//
//    thrust::device_vector<T> to_device(const T &host_data) const {
//        // Implementation of conversion from host to device
//        spdlog::debug("Converted pydin_array object from host to device.");
//    }
//
//    // Other necessary methods...
//
//private:
//    std::vector<T>* host_data_;
//    thrust::device_vector<T> *device_data_;
//};
//
//py::class_<pydin_array<P>>(m, "point_array")
//        .def(py::init<>())
//        .def(py::init<const P &>())
//        .def("to_host", &pydin_array<P>::to_host)
//        .def("to_device", &pydin_array<P>::to_device)
//        .def(
//                "__getitem__",
//                [](const pydin_array<P> &s, size_t i) {
//                    if (i < s.size()) return s[i];
//                    else
//                        throw py::index_error();
//                },
//                py::return_value_policy::copy)
//        .def("__setitem__",
//             [](pydin_array<P> &s, size_t i, const P &v) {
//                 if (i < s.size()) s[i] = v;
//                 else
//                     throw py::index_error();
//             })
//        .def("__len__", &pydin_array<P>::size);
#include <odin/eigen.h>
#include <odin/models/shape/algorithms.h>
#include <odin/models/shape/mesh/marching_cubes.h>

template<typename U>
void bind_algorithms(pybind11::module_ &m) {
    using namespace odin::shape;
    using namespace odin::algorithms;
    using namespace pybind11::literals;

    using point_type = Eigen::Matrix<U, 3, 1>;
    py::class_<Voxel<point_type>>(m, "Voxel")
            .def(py::init<>())
            .def(py::init<const point_type &, U>())
            .def_readwrite("position", &Voxel<point_type>::position)
            .def_readwrite("value", &Voxel<point_type>::value)
            .def_readwrite("is_active", &Voxel<point_type>::is_active);

    py::class_<mesh::face_vertex<U, point_type, 3>>(m, "face_vertex")
            //            .def(py::init<>())
            //            .def(py::init<const point_type &, const point_type &, const point_type &>())
            .def("get_vertices", &mesh::face_vertex<U, point_type, 3>::get_vertices)
            .def("get_facets", &mesh::face_vertex<U, point_type, 3>::get_facets);


    m.def(
            "marching_cubes",
            [](const std::array<Eigen::Tensor<U, 3>, 3> &grid_positions,
               const Eigen::Tensor<U, 3>                &grid_values,
               U isovalue) { return marching_cubes(grid_positions, grid_values, isovalue); },
            "grid_positions"_a,
            "grid_values"_a,
            "iso_value"_a);


//    m.def("marching_cubes",
//          &marching_cubes<U, 3>,
//          "grid_values"_a,
//          "iso_value"_a,
//          "grid_scale"_a = Eigen::Matrix<U, 3, 1>::Ones());
//

    m.def(
            "ellipse_of_intersection",
            [](U nx, U ny, U nz, U f, U a, U b, U c) {
                return ellipse_of_intersection(nx, ny, nz, f, a, b, c);
            },
            "nx"_a,// x-component of normal vector
            "ny"_a,// y-component of normal vector
            "nz"_a,// z-component of normal vector
            "f"_a, // plane offset
            "a"_a, // x-axis ellipsoid radius
            "b"_a, // y-axis ellipsoid radius
            "c"_a  // z-axis ellipsoid radius
    );

    m.def(
            "ellipse_of_intersection",
            [](vec3_type<U> n, U f, U a, U b, U c) {
                return ellipse_of_intersection(n, f, a, b, c);
            },
            "n"_a,// normal vector
            "f"_a,// plane offset
            "a"_a,// x-axis ellipsoid radius
            "b"_a,// y-axis ellipsoid radius
            "c"_a // z-axis ellipsoid radius
    );

    m.def(
            "ellipse_of_intersection",
            [](vec3_type<U> n, U f, vec3_type<U> r) { return ellipse_of_intersection(n, f, r); },
            "n"_a,// normal vector
            "f"_a,// plane offset
            "r"_a // ellipsoid radii
    );

    m.def(
            "calculate_polar_plane_pole",
            [](vec3_type<U> n, U f, vec3_type<U> r) {
                return calculate_polar_plane_pole(n, f, r);
            },
            "n"_a,// normal vector
            "f"_a,// plane offset
            "r"_a // ellipsoid radii
    );

    m.def(
            "calculate_polar_plane_pole",
            [](vec3_type<U> n, U f, U a, U b, U c) {
                return calculate_polar_plane_pole(n, f, a, b, c);
            },
            "n"_a,// normal vector
            "f"_a,// plane offset
            "a"_a,// x-axis ellipsoid radius
            "b"_a,// y-axis ellipsoid radius
            "c"_a // z-axis ellipsoid radius
    );

    m.def(
            "calculate_polar_plane_pole",
            [](U nx, U ny, U nz, U f, U a, U b, U c) {
                return calculate_polar_plane_pole(nx, ny, nz, f, a, b, c);
            },
            "nx"_a,// x-component of normal vector
            "ny"_a,// y-component of normal vector
            "nz"_a,// z-component of normal vector
            "f"_a, // plane offset
            "a"_a, // x-axis ellipsoid radius
            "b"_a, // y-axis ellipsoid radius
            "c"_a  // z-axis ellipsoid radius
    );
}


namespace pydin {
    using namespace odin::shape;
    using namespace odin;

    template<typename T>
    py::class_<T> bind_derived_shape(pybind11::module_ &m, const std::string &shape_name) {
        using point_type        = typename T::point_type;
        using point_series_type = point_series_type_trait<T>::type;
        return py::class_<T>(m, shape_name.c_str())
                .def("volume", &T::volume)
                .def("surface_area", &T::surface_area)
                .def("is_inside",
                     [](T &instance, const typename T::point_type &point) {
                         return instance.is_inside(point);
                     })
                .def("is_inside",
                     [](T &instance, const typename T::point_series_type &points) {
                         return instance.is_inside(points);
                     })
                .def("centroid", &T::centroid)
                .def("get_center", &T::get_center)
                .def("set_center", &T::set_center)
                .def("ray_intersection", &T::ray_intersection, "origin"_a, "direction"_a)
                .def("ray_intersection_series",
                     &T::ray_intersection_series,
                     "origins"_a,
                     "directions"_a)
                .def(
                        "ray_intersection_series2",
                        [](T                  &instance,
                           py::array_t<double> origins_py,
                           py::array_t<double> directions_py) {
                            // Convert the Python arrays to C++ Eigen matrices
                            Eigen::MatrixXd origins_eigen = py::cast<Eigen::MatrixXd>(origins_py);
                            Eigen::MatrixXd directions_eigen
                                    = py::cast<Eigen::MatrixXd>(directions_py);

                            // Convert Eigen matrices to thrust::device_vector
                            thrust::device_vector<Eigen::Matrix<double, 3, 1>> origins_thrust,
                                    directions_thrust;
                            for (int i = 0; i < origins_eigen.cols(); i++) {
                                origins_thrust.push_back(origins_eigen.col(i));
                                directions_thrust.push_back(directions_eigen.col(i));
                            }

                            // Invoke the method
                            thrust::device_vector<Eigen::Matrix<double, 3, 1>> result_device
                                    = instance.ray_intersection_series2(origins_thrust,
                                                                        directions_thrust);

                            // Convert the result from thrust::device_vector to thrust::host_vector
                            thrust::host_vector<Eigen::Matrix<double, 3, 1>> result_host
                                    = result_device;

                            // Convert the result from thrust::host_vector to Eigen::MatrixXd
                            Eigen::MatrixXd result_eigen(3, result_host.size());
                            for (size_t i = 0; i < result_host.size(); ++i) {
                                result_eigen.col(i) = result_host[i];
                            }

                            // Return the result as a Python object
                            return py::cast(result_eigen);
                        },
                        "origins"_a,
                        "directions"_a);
    }

    template<typename U, Point P>
    void bind_shape(py::module_ &m) {

        bind_algorithms<U>(m);

        using sphere_type = sphere<U, P>;
        bind_derived_shape<sphere_type>(m, "sphere")
                .def(pybind11::init<U, P>(), "radius"_a, "center"_a = zero<P>())
                .def("get_radius", &sphere_type::get_radius)
                .def("set_radius", &sphere_type::set_radius)
                .def_property("radius", &sphere_type::get_radius, &sphere_type::set_radius);


        using ellipsoid_type = ellipsoid<U, P>;
        bind_derived_shape<ellipsoid_type>(m, "ellipsoid")
                .def(pybind11::init<U, U, U, P>(),
                     "radius_a"_a,
                     "radius_b"_a,
                     "radius_c"_a,
                     "center"_a = zero<P>())
                .def(pybind11::init<P, P>(), "radii"_a, "center"_a = zero<P>())
                .def("get_radius_a", &ellipsoid_type::get_radius_a)
                .def("set_radius_a", &ellipsoid_type::set_radius_a, "radius"_a)
                .def("get_radius_b", &ellipsoid_type::get_radius_b)
                .def("set_radius_b", &ellipsoid_type::set_radius_b, "radius"_a)
                .def("get_radius_c", &ellipsoid_type::get_radius_c)
                .def("set_radius_c", &ellipsoid_type::set_radius_c, "radius"_a)
                .def("get_silhouette_dimensions",
                     &ellipsoid_type::get_silhouette_dimensions,
                     "source"_a,
                     "direction"_a)
                .def_property("radius_a",
                              &ellipsoid_type::get_radius_a,
                              &ellipsoid_type::set_radius_a)
                .def_property("radius_b",
                              &ellipsoid_type::get_radius_b,
                              &ellipsoid_type::set_radius_b)
                .def_property("radius_c",
                              &ellipsoid_type::get_radius_c,
                              &ellipsoid_type::set_radius_c)
                .def("get_radii", &ellipsoid_type::get_radii)
                .def("set_radii", &ellipsoid_type::set_radii, "radii"_a)
                .def_property("radii", &ellipsoid_type::get_radii, &ellipsoid_type::set_radii);
    }

}// namespace pydin

#endif// PYDIN_BIND_SHAPE_H
