#include <pyuipc/constitution/module.h>
#include <pyuipc/constitution/constitution.h>
#include <pyuipc/constitution/elastic_moduli.h>
#include <pyuipc/constitution/finite_element_constitution.h>
#include <pyuipc/constitution/particle.h>
#include <pyuipc/constitution/hookean_spring.h>
#include <pyuipc/constitution/shell_neo_hookean.h>
#include <pyuipc/constitution/stable_neo_hookean.h>
#include <pyuipc/constitution/affine_body_constitution.h>
#include <pyuipc/constitution/constraint.h>
#include <pyuipc/constitution/soft_position_constraint.h>
#include <pyuipc/constitution/finite_element_extra_constitution.h>
#include <pyuipc/constitution/kirchhoff_rod_bending.h>

namespace pyuipc::constitution
{
Module::Module(py::module& m)
{
    PyConstitution{m};
    PyConstraint{m};
    PyElasticModuli{m};

    // Finite Element Constitutions
    PyFiniteElementConstitution{m};
    PyParticle{m};
    PyHookeanSpring{m};
    PyShellNeoHookean{m};
    PyStableNeoHookean{m};

    // Finite Extra Constitutions
    PyFiniteElementExtraConstitution{m};
    PyKirchhoffRodBending{m};

    // Affine Body Constitutions
    PyAffineBodyConstitution{m};

    // Constraints
    PySoftPositionConstraint{m};
}
}  // namespace pyuipc::constitution
