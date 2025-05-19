#include <pyuipc/constitution/discrete_pattern_bending.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/discrete_pattern_bending.h>
#include <pyuipc/common/json.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> 
#include <memory_resource>  

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyDiscretePatternBending::PyDiscretePatternBending(py::module& m)
{
    namespace py = pybind11;
    auto cls = py::class_<DiscretePatternBending,
                          FiniteElementExtraConstitution>(m, "DiscretePatternBending");

    cls.def(py::init<const Json&>(),
            py::arg("config") = DiscretePatternBending::default_config());

    cls.def_static("default_config", &DiscretePatternBending::default_config);

    /* 旧接口：list/tuple 也能用，但效率差                   *
     * 新接口：接受 numpy.ndarray，连贯内存→vector<Float>   */
    cls.def(
        "apply_to",
        [](DiscretePatternBending& self,
           geometry::SimplicialComplex& sc,
           py::array_t<Float,
                       py::array::c_style | py::array::forcecast> rest_angle,
           Float E)
        {
            auto buf = rest_angle.request();
            if (buf.ndim != 1)
                throw std::runtime_error("rest_angle 必须是一维数组");
    
            const Float* beg = static_cast<const Float*>(buf.ptr);
            const Float* end = beg + buf.shape[0];
    
            /* ② 用缺省内存资源构造 pmr::vector */
            std::pmr::vector<Float> angles{beg, end};
    
            /* ③ 传 pmr::vector 给 apply_to */
            self.apply_to(sc, angles, E);
        },
        py::arg("sc"),
        py::arg("rest_angle"),
        py::arg("E") = static_cast<Float>(100.0)
    );
}
}  // namespace pyuipc::constitution
