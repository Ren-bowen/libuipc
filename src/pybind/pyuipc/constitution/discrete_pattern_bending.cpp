#include <pyuipc/constitution/discrete_pattern_bending.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/discrete_pattern_bending.h>
#include <pyuipc/common/json.h>
#include <pybind11/stl.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyDiscretePatternBending::PyDiscretePatternBending(py::module& m)
{
    auto class_DiscretePatternBending =
        py::class_<DiscretePatternBending, FiniteElementExtraConstitution>(m, "DiscretePatternBending");

    class_DiscretePatternBending.def(py::init<const Json&>(),
                                   py::arg("config") =
                                       DiscretePatternBending::default_config());

    class_DiscretePatternBending.def_static("default_config",
                                          &DiscretePatternBending::default_config);

    class_DiscretePatternBending.def(
        "apply_to", &DiscretePatternBending::apply_to, py::arg("sc"), py::arg("rest_angle"), py::arg("E") = 100.0_kPa);
}
}  // namespace pyuipc::constitution
