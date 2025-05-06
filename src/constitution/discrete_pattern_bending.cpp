#include <uipc/constitution/discrete_pattern_bending.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>

namespace uipc::constitution
{
constexpr U64 DiscretePatternBendingUID = 67;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = DiscretePatternBendingUID;
    info.name = "DiscretePatternBending";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

DiscretePatternBending::DiscretePatternBending(const Json& json)
    : m_config{json}
{
}

void DiscretePatternBending::apply_to(geometry::SimplicialComplex& sc,
                                   const vector<Float>&            rest_angles,
                                   Float                           E)
{
    Base::apply_to(sc);
    auto bs = sc.edges().find<Float>("bending_stiffness");
    if(!bs)
    {
        bs = sc.edges().create<Float>("bending_stiffness");
    }
    auto bs_view = geometry::view(*bs);
    std::ranges::fill(bs_view, E);

    auto rest_angles_attr = sc.edges().find<Float>("rest_angle");
    if (!rest_angles_attr)
        rest_angles_attr = sc.edges().create<Float>("rest_angle");
    auto dst = view(*rest_angles_attr);  // span<Float>
    UIPC_ASSERT(dst.size() == rest_angles.size(),
                "rest angle size mismatch");
    for (size_t i = 0; i < rest_angles.size(); ++i)
        dst[i] = rest_angles[i];
}

U64 DiscretePatternBending::get_uid() const noexcept
{
    return DiscretePatternBendingUID;
}

Json DiscretePatternBending::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution
