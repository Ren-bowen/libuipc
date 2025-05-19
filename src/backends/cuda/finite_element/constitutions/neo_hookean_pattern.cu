#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/neo_hookean_pattern_function.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <muda/ext/eigen/inverse.h>
#include <utils/codim_thickness.h>
#include <utils/matrix_assembly_utils.h>

namespace uipc::backend::cuda
{
class NeoHookeanPattern final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 66;

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_kappas;
    vector<Float> h_lambdas;
    vector<Float> h_X_rests_1;
    vector<Float> h_X_rests_2;
    vector<Float> h_X_rests_3;
    vector<Float> h_X_rests_4;
    vector<Float> h_X_rests_5;
    vector<Float> h_X_rests_6;
    vector<Float> h_X_rests_7;
    vector<Float> h_X_rests_8;
    vector<Float> h_X_rests_9;

    muda::DeviceBuffer<Float> kappas;
    muda::DeviceBuffer<Float> lambdas;
    muda::DeviceBuffer<Float> X_rests_1;
    muda::DeviceBuffer<Float> X_rests_2;
    muda::DeviceBuffer<Float> X_rests_3;
    muda::DeviceBuffer<Float> X_rests_4;
    muda::DeviceBuffer<Float> X_rests_5;
    muda::DeviceBuffer<Float> X_rests_6;
    muda::DeviceBuffer<Float> X_rests_7;
    muda::DeviceBuffer<Float> X_rests_8;
    muda::DeviceBuffer<Float> X_rests_9;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();
        h_kappas.resize(N);
        h_lambdas.resize(N);
        h_X_rests_1.resize(N);
        h_X_rests_2.resize(N);
        h_X_rests_3.resize(N);
        h_X_rests_4.resize(N);
        h_X_rests_5.resize(N);
        h_X_rests_6.resize(N);
        h_X_rests_7.resize(N);
        h_X_rests_8.resize(N);
        h_X_rests_9.resize(N);
        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) {
                auto mu      = sc.triangles().find<Float   >("mu");
                auto lambda  = sc.triangles().find<Float   >("lambda");
                auto X_rest_1  = sc.triangles().find<Float >("X_rests_1");
                auto X_rest_2  = sc.triangles().find<Float >("X_rests_2");
                auto X_rest_3  = sc.triangles().find<Float >("X_rests_3");
                auto X_rest_4  = sc.triangles().find<Float >("X_rests_4");
                auto X_rest_5  = sc.triangles().find<Float >("X_rests_5");
                auto X_rest_6  = sc.triangles().find<Float >("X_rests_6");
                auto X_rest_7  = sc.triangles().find<Float >("X_rests_7");
                auto X_rest_8  = sc.triangles().find<Float >("X_rests_8");
                auto X_rest_9  = sc.triangles().find<Float >("X_rests_9");
                return zip(mu->view(), lambda->view(), X_rest_1->view(), X_rest_2->view(), X_rest_3->view(),
                           X_rest_4->view(), X_rest_5->view(), X_rest_6->view(), X_rest_7->view(),
                           X_rest_8->view(), X_rest_9->view());
                // return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto list) {
                std::size_t v = I.global_index();
                auto&& [mu, lambda, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9] = list;
                // auto&& [mu, lambda] = triple;
        
                h_kappas [v] = mu;
                h_lambdas[v] = lambda;
                h_X_rests_1[v] = X_1;
                h_X_rests_2[v] = X_2;
                h_X_rests_3[v] = X_3;
                h_X_rests_4[v] = X_4;
                h_X_rests_5[v] = X_5;
                h_X_rests_6[v] = X_6;
                h_X_rests_7[v] = X_7;
                h_X_rests_8[v] = X_8;
                h_X_rests_9[v] = X_9;
            });
        kappas.resize(N);
        kappas.view().copy_from(h_kappas.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());

        X_rests_1.resize(N);
        X_rests_1.view().copy_from(h_X_rests_1.data());
        X_rests_2.resize(N);
        X_rests_2.view().copy_from(h_X_rests_2.data());
        X_rests_3.resize(N);
        X_rests_3.view().copy_from(h_X_rests_3.data());
        X_rests_4.resize(N);
        X_rests_4.view().copy_from(h_X_rests_4.data());
        X_rests_5.resize(N);
        X_rests_5.view().copy_from(h_X_rests_5.data());
        X_rests_6.resize(N);
        X_rests_6.view().copy_from(h_X_rests_6.data());
        X_rests_7.resize(N);
        X_rests_7.view().copy_from(h_X_rests_7.data());
        X_rests_8.resize(N);
        X_rests_8.view().copy_from(h_X_rests_8.data());
        X_rests_9.resize(N);
        X_rests_9.view().copy_from(h_X_rests_9.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace NH = sym::pattern_neo_hookean;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus        = kappas.cviewer().name("mus"),
                    lambdas    = lambdas.cviewer().name("lambdas"),
                    rest_areas = info.rest_areas().viewer().name("rest_area"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    // X_bars   = info.x_bars().viewer().name("x_bars"),
                    // X_bars   = X_rests.cviewer().name("X_bars"),
                    X_bars_1 = X_rests_1.cviewer().name("X_bars_1"),
                    X_bars_2 = X_rests_2.cviewer().name("X_bars_2"),
                    X_bars_3 = X_rests_3.cviewer().name("X_bars_3"),
                    X_bars_4 = X_rests_4.cviewer().name("X_bars_4"),
                    X_bars_5 = X_rests_5.cviewer().name("X_bars_5"),
                    X_bars_6 = X_rests_6.cviewer().name("X_bars_6"),
                    X_bars_7 = X_rests_7.cviewer().name("X_bars_7"),
                    X_bars_8 = X_rests_8.cviewer().name("X_bars_8"),
                    X_bars_9 = X_rests_9.cviewer().name("X_bars_9"),
                    dt       = info.dt()] __device__(int I)
                   {
                       Vector9  X;
                       Vector3i idx = indices(I);
                       for(int i = 0; i < 3; ++i)
                           X.segment<3>(3 * i) = xs(idx(i));

                       Vector9 X_bar;
                       X_bar(0) = X_bars_1(I);
                       X_bar(1) = X_bars_2(I);
                       X_bar(2) = X_bars_3(I);
                       X_bar(3) = X_bars_4(I);
                       X_bar(4) = X_bars_5(I);
                       X_bar(5) = X_bars_6(I);
                       X_bar(6) = X_bars_7(I);
                       X_bar(7) = X_bars_8(I);
                       X_bar(8) = X_bars_9(I);
                    //    Vector9 X_bar;
                    //    for(int i = 0; i < 3; ++i)
                    //        X_bar.segment<3>(3 * i) = X_bars(idx(i));
                       Matrix2x2 IB;
                       NH::A(IB, X_bar);
                       IB = muda::eigen::inverse(IB);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Matrix2x2 A;
                           NH::A(A, X);
                           Float detA = A.determinant();
                       }

                       Float mu        = mus(I);
                       Float lambda    = lambdas(I);
                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float E;
                       NH::E(E, mu, lambda, X, IB);
                       energies(I) = E * rest_area * thickness * dt * dt;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace NH = sym::pattern_neo_hookean;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus     = kappas.cviewer().name("mus"),
                    lambdas = lambdas.cviewer().name("lambdas"),
                    indices = info.indices().viewer().name("indices"),
                    xs      = info.xs().viewer().name("xs"),
                    // X_bars  = info.x_bars().viewer().name("x_bars"),
                    // X_bars   = X_rests.cviewer().name("X_bars"),
                    X_bars_1 = X_rests_1.cviewer().name("X_bars_1"),
                    X_bars_2 = X_rests_2.cviewer().name("X_bars_2"),
                    X_bars_3 = X_rests_3.cviewer().name("X_bars_3"),
                    X_bars_4 = X_rests_4.cviewer().name("X_bars_4"),
                    X_bars_5 = X_rests_5.cviewer().name("X_bars_5"),
                    X_bars_6 = X_rests_6.cviewer().name("X_bars_6"),
                    X_bars_7 = X_rests_7.cviewer().name("X_bars_7"),
                    X_bars_8 = X_rests_8.cviewer().name("X_bars_8"),
                    X_bars_9 = X_rests_9.cviewer().name("X_bars_9"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    G3s        = info.gradients().viewer().name("gradient"),
                    H3x3s      = info.hessians().viewer().name("hessian"),
                    rest_areas = info.rest_areas().viewer().name("volumes"),
                    dt         = info.dt()] __device__(int I) mutable
                   {
                       Vector9  X;
                       Vector3i idx = indices(I);
                       for(int i = 0; i < 3; ++i)
                           X.segment<3>(3 * i) = xs(idx(i));

                       Vector9 X_bar;
                       X_bar(0) = X_bars_1(I);
                       X_bar(1) = X_bars_2(I);
                       X_bar(2) = X_bars_3(I);
                       X_bar(3) = X_bars_4(I);
                       X_bar(4) = X_bars_5(I);
                       X_bar(5) = X_bars_6(I);
                       X_bar(6) = X_bars_7(I);
                       X_bar(7) = X_bars_8(I);
                       X_bar(8) = X_bars_9(I);
                    //    Vector9 X_bar;
                    //    for(int i = 0; i < 3; ++i)
                    //        X_bar.segment<3>(3 * i) = X_bars(idx(i));

                       Matrix2x2 IB;
                       NH::A(IB, X_bar);
                       IB = muda::eigen::inverse(IB);

                       Float mu        = mus(I);
                       Float lambda    = lambdas(I);
                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float Vdt2 = rest_area * thickness * dt * dt;

                       Vector9 G;
                       NH::dEdX(G, mu, lambda, X, IB);
                       G *= Vdt2;
                       assemble<3>(G3s, I * 3, idx, G);

                       Matrix9x9 H;
                       NH::ddEddX(H, mu, lambda, X, IB);
                       H *= Vdt2;
                       make_spd(H);
                       assemble<3>(H3x3s, I * 3 * 3, idx, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(NeoHookeanPattern);
}  // namespace uipc::backend::cuda
