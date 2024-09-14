#pragma once
#include <animator/animator.h>
#include <finite_element/finite_element_method.h>
#include <gradient_hessian_computer.h>
#include <line_search/line_searcher.h>

namespace uipc::backend::cuda
{
class FiniteElementConstraint;
class FiniteElementAnimator final : public Animator
{
  public:
    using Animator::Animator;

    using AnimatedGeoInfo = FiniteElementMethod::GeoInfo;

    class Impl;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        span<const AnimatedGeoInfo> animated_geo_infos() const;

        SizeT anim_vertex_count() const noexcept;

        span<const IndexT> anim_indices() const;

        template <typename ForEach, typename ViewGetter>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetter&&                    view_getter,
                      ForEach&&                       for_each_action) noexcept;

      private:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, SizeT index, Float dt)
            : m_impl(impl)
            , m_index(index)
            , m_dt(dt)
        {
        }

        Float                      dt() const noexcept;
        muda::CBufferView<Vector3> xs() const noexcept;
        muda::CBufferView<Float>   masses() const noexcept;
        muda::CBufferView<IndexT>  is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        muda::BufferView<Float> energies() const noexcept;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        muda::DoubletVectorView<Float, 3> gradients() const noexcept;
        muda::TripletMatrixView<Float, 3> hessians() const noexcept;
    };

    class ReportExtentInfo
    {
      public:
        void hessian_block_count(SizeT count) noexcept;
        void gradient_segment_count(SizeT count) noexcept;
        void energy_count(SizeT count) noexcept;

      private:
        friend class FiniteElementAnimator;
        SizeT m_hessian_block_count    = 0;
        SizeT m_gradient_segment_count = 0;
        SizeT m_energy_count           = 0;
    };

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);
        void step();

        Float dt;

        FiniteElementMethod* finite_element_method = nullptr;
        SimSystemSlotCollection<FiniteElementConstraint> constraints;
        unordered_map<U64, SizeT> uid_to_constraint_index;

        vector<AnimatedGeoInfo> anim_geo_infos;
        vector<SizeT>           constraint_geo_info_offsets;
        vector<SizeT>           constraint_geo_info_counts;

        vector<SizeT> constraint_vertex_counts;
        vector<SizeT> constraint_vertex_offsets;

        vector<IndexT> h_anim_indices;
        muda::DeviceBuffer<IndexT> anim_indices;  // view(anim_vertex_offsets[index], anim_vertex_counts[index])

        // Constraints
        muda::DeviceVar<Float> constraint_energy;  // Constraint Energy
        muda::DeviceBuffer<Float> constraint_energies;  // Constraint Energy Per Element
        vector<SizeT> constraint_energy_offsets;
        vector<SizeT> constraint_energy_counts;

        muda::DeviceDoubletVector<Float, 3> constraint_gradient;  // Constraint Gradient Per Vertex
        vector<SizeT> constraint_gradient_offsets;
        vector<SizeT> constraint_gradient_counts;

        muda::DeviceTripletMatrix<Float, 3> constraint_hessian;  // Constraint Hessian Per Vertex
        vector<SizeT> constraint_hessian_offsets;
        vector<SizeT> constraint_hessian_counts;
    };

  private:
    friend class FiniteElementConstraint;
    void add_constraint(FiniteElementConstraint* constraint);  // only be called by FiniteElementConstraint

    friend class FEMLineSearchReporter;
    Float compute_energy(LineSearcher::EnergyInfo& info);  // only be called by FEMLineSearchReporter

    friend class FEMGradientHessianComputer;
    void compute_gradient_hessian(GradientHessianComputer::ComputeInfo& info);  // only be called by GradientHessianComputer

    friend class FEMLinearSubsystem;
    class ExtentInfo
    {
      public:
        SizeT hessian_block_count;
    };
    void report_extent(ExtentInfo& info);  // only be called by FEMLinearSubsystem
    class AssembleInfo
    {
      public:
        muda::TripletMatrixView<Float, 3> hessians;
        muda::DoubletVectorView<Float, 3> gradients;
    };
    void assemble(AssembleInfo& info);  // only be called by FEMLinearSubsystem

    Impl m_impl;

    virtual void do_init() override;
    virtual void do_step() override;
    virtual void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::cuda


#include "details/finite_element_animator.inl"