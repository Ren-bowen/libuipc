#pragma once
#include <sim_system.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <functional>
#include <Eigen/Geometry>
#include <collision_detection/aabb.h>

namespace uipc::backend::cuda
{
class VertexReporter;
class GlobalVertexManager : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class VertexCountInfo
    {
      public:
        void count(SizeT count) noexcept;
        void changable(bool is_changable) noexcept;

      private:
        friend class GlobalVertexManager;
        SizeT m_count;
        bool  m_changable;
    };

    class VertexAttributeInfo
    {
      public:
        VertexAttributeInfo(Impl* impl, SizeT index) noexcept;
        muda::BufferView<IndexT>  coindex() const noexcept;
        muda::BufferView<Vector3> positions() const noexcept;

      private:
        friend class GlobalVertexManager;
        SizeT m_index;
        Impl* m_impl;
    };

    class VertexDisplacementInfo
    {
      public:
        VertexDisplacementInfo(Impl* impl, SizeT index) noexcept;
        muda::BufferView<Vector3> displacements() const noexcept;
        muda::CBufferView<IndexT> coindex() const noexcept;

      private:
        friend class GlobalVertexManager;
        SizeT m_index;
        Impl* m_impl;
    };

    class Impl;

    class VertexRegister
    {
      public:
        VertexRegister(std::string_view name,
                       std::function<void(VertexCountInfo&)>&& report_vertex_count,
                       std::function<void(VertexAttributeInfo&)>&& report_vertex_attributes,
                       std::function<void(VertexDisplacementInfo&)>&& report_vertex_displacement) noexcept;

      private:
        friend class GlobalVertexManager;
        friend class Impl;

        std::string                               m_name;
        std::function<void(VertexCountInfo&)>     m_report_vertex_count;
        std::function<void(VertexAttributeInfo&)> m_report_vertex_attributes;
        std::function<void(VertexDisplacementInfo&)> m_report_vertex_displacement;
    };

    muda::CBufferView<IndexT>  coindex() const noexcept;
    muda::CBufferView<Vector3> positions() const noexcept;
    muda::CBufferView<Vector3> displacements() const noexcept;

    void add_reporter(VertexReporter* reporter);

    AABB vertex_bounding_box() const noexcept;

  public:
    class Impl
    {
        friend class GlobalVertexManager;

      public:
        Impl() = default;
        void  init_vertex_info();
        void  rebuild_vertex_info();
        Float compute_max_displacement();
        AABB  compute_vertex_bounding_box();
        void  collect_vertex_displacements();

        template <typename T>
        muda::BufferView<T> subview(muda::DeviceBuffer<T>& buffer, SizeT index) const noexcept;

      private:
        /**
         * @brief A mapping from the global vertex index to the coindex.
         * 
         * The values of coindex is dependent on the reporters, which can be:
         * 1) the local index of the vertex
         * 2) or any other information that is needed to be stored.
         */
        muda::DeviceBuffer<IndexT>  coindex;
        muda::DeviceBuffer<Vector3> positions;
        muda::DeviceBuffer<Vector3> displacements;

        muda::DeviceVar<Float>   max_disp;
        muda::DeviceVar<Vector3> min_pos;
        muda::DeviceVar<Vector3> max_pos;

        list<VertexReporter*>   vertex_reporter_buffer;
        vector<VertexReporter*> vertex_reporters;
        vector<VertexReporter*> changable_vertex_reporters;

        vector<SizeT> reporter_vertex_offsets;
        vector<SizeT> reporter_vertex_counts;

        AABB vertex_bounding_box;
    };

  private:
    friend class SimEngine;
    void  init_vertex_info();
    void  rebuild_vertex_info();
    Float compute_max_displacement();
    AABB  compute_vertex_bounding_box();
    Impl  m_impl;
};
}  // namespace uipc::backend::cuda

#include "details/global_vertex_manager.inl"