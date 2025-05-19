#include <uipc/constitution/neo_hookean_pattern.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/constitution/conversion.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid = 66, .name = "NeoHookeanPattern", .type = string{builtin::FiniteElement}});
    return uids;
}

NeoHookeanPattern::NeoHookeanPattern(const Json& config) noexcept
    : m_config(config)
{
}

void NeoHookeanPattern::apply_to(geometry::SimplicialComplex& sc,
                               const vector<Float>&         X_bars,
                               const ElasticModuli&         moduli,
                               Float                        mass_density,
                               Float                        thickness
                            ) const
{
    Base::apply_to(sc, mass_density, thickness);
    size_t nTri = sc.triangles().size();
    UIPC_ASSERT(sc.dim() == 2, "NeoHookeanPattern only supports 2D simplicial complex");
    UIPC_ASSERT(9 * nTri == X_bars.size(),
                "Rest shape size mismatch");

    const Float mu     = moduli.mu();
    const Float lambda = moduli.lambda();

    auto mu_attr = sc.triangles().find<Float>("mu");
    if (!mu_attr)
        mu_attr = sc.triangles().create<Float>("mu", mu);
    std::ranges::fill(geometry::view(*mu_attr), mu);

    auto lambda_attr = sc.triangles().find<Float>("lambda");
    if (!lambda_attr)
        lambda_attr = sc.triangles().create<Float>("lambda", lambda);
    std::ranges::fill(geometry::view(*lambda_attr), lambda);
    
    // auto X_bars_attr = sc.triangles().find<Vector9>("X_rests");
    // if (!X_bars_attr)
    //     X_bars_attr = sc.triangles().create<Vector9>("X_rests");
    // auto dst = view(*X_bars_attr);           // span<Vector9>
    // UIPC_ASSERT(dst.size() == nTri,
    //             "attribute size mismatch");

    // for (size_t i = 0; i < nTri; ++i)
    //     for (size_t j = 0; j < 9; ++j)
    //         dst[i][j] = X_bars[i * 9 + j];

    auto X_bars_1_attr = sc.triangles().find<Float>("X_rests_1");
    if (!X_bars_1_attr)
        X_bars_1_attr = sc.triangles().create<Float>("X_rests_1");
    auto X_bars_2_attr = sc.triangles().find<Float>("X_rests_2");
    if (!X_bars_2_attr)
        X_bars_2_attr = sc.triangles().create<Float>("X_rests_2");
    auto X_bars_3_attr = sc.triangles().find<Float>("X_rests_3");
    if (!X_bars_3_attr)
        X_bars_3_attr = sc.triangles().create<Float>("X_rests_3");
    auto X_bars_4_attr = sc.triangles().find<Float>("X_rests_4");
    if (!X_bars_4_attr)
        X_bars_4_attr = sc.triangles().create<Float>("X_rests_4");
    auto X_bars_5_attr = sc.triangles().find<Float>("X_rests_5");
    if (!X_bars_5_attr)
        X_bars_5_attr = sc.triangles().create<Float>("X_rests_5");
    auto X_bars_6_attr = sc.triangles().find<Float>("X_rests_6");
    if (!X_bars_6_attr)
        X_bars_6_attr = sc.triangles().create<Float>("X_rests_6");
    auto X_bars_7_attr = sc.triangles().find<Float>("X_rests_7");
    if (!X_bars_7_attr)
        X_bars_7_attr = sc.triangles().create<Float>("X_rests_7");
    auto X_bars_8_attr = sc.triangles().find<Float>("X_rests_8");
    if (!X_bars_8_attr)
        X_bars_8_attr = sc.triangles().create<Float>("X_rests_8");
    auto X_bars_9_attr = sc.triangles().find<Float>("X_rests_9");
    if (!X_bars_9_attr)
        X_bars_9_attr = sc.triangles().create<Float>("X_rests_9");
    auto dst_1 = view(*X_bars_1_attr);           
    auto dst_2 = view(*X_bars_2_attr);        
    auto dst_3 = view(*X_bars_3_attr);         
    auto dst_4 = view(*X_bars_4_attr);
    auto dst_5 = view(*X_bars_5_attr);
    auto dst_6 = view(*X_bars_6_attr);
    auto dst_7 = view(*X_bars_7_attr);
    auto dst_8 = view(*X_bars_8_attr);
    auto dst_9 = view(*X_bars_9_attr);
    for (size_t i = 0; i < nTri; ++i)
    {
        dst_1[i] = X_bars[i * 9 + 0];
        dst_2[i] = X_bars[i * 9 + 1];
        dst_3[i] = X_bars[i * 9 + 2];
        dst_4[i] = X_bars[i * 9 + 3];
        dst_5[i] = X_bars[i * 9 + 4];
        dst_6[i] = X_bars[i * 9 + 5];
        dst_7[i] = X_bars[i * 9 + 6];
        dst_8[i] = X_bars[i * 9 + 7];
        dst_9[i] = X_bars[i * 9 + 8];
        // printf("i = %d\n", i);
        // printf("dist_1[%d] = %f\n", i, dst_1[i]);
        // printf("dist_2[%d] = %f\n", i, dst_2[i]);
        // printf("dist_3[%d] = %f\n", i, dst_3[i]);
        // printf("dist_4[%d] = %f\n", i, dst_4[i]);
        // printf("dist_5[%d] = %f\n", i, dst_5[i]);
        // printf("dist_6[%d] = %f\n", i, dst_6[i]);
        // printf("dist_7[%d] = %f\n", i, dst_7[i]);
        // printf("dist_8[%d] = %f\n", i, dst_8[i]);
        // printf("dist_9[%d] = %f\n", i, dst_9[i]);
    }

}

Json NeoHookeanPattern::default_config() noexcept
{
    return Json::object();
}

U64 NeoHookeanPattern::get_uid() const noexcept
{
    return 66;
}
}  // namespace uipc::constitution
