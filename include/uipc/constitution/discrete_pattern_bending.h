#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API DiscretePatternBending : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    DiscretePatternBending(const Json& json = default_config());

    void apply_to(geometry::SimplicialComplex& sc, const vector<Float>& rest_angles, Float E = 100.0_kPa);

    static Json default_config();


  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
