#pragma once
#include <collision_detection/vertex_half_plane_trajectory_filter.h>

namespace uipc::backend::cuda
{
void VertexHalfPlaneTrajectoryFilter::do_build()
{
    m_impl.global_vertex_manager = &require<GlobalVertexManager>();
    m_impl.global_simplicial_surface_manager = &require<GlobalSimpicialSurfaceManager>();
    m_impl.global_contact_manager = &require<GlobalContactManager>();
    m_impl.half_plane             = &require<HalfPlane>();
    auto global_trajectory_filter = &require<GlobalTrajectoryFilter>();

    BuildInfo info;
    do_build(info);

    global_trajectory_filter->add_filter(this);
}

void VertexHalfPlaneTrajectoryFilter::do_detect(GlobalTrajectoryFilter::DetectInfo& info)
{
    DetectInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();

    do_detect(this_info);  // call the derived class implementation
}

void VertexHalfPlaneTrajectoryFilter::do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info)
{
    FilterActiveInfo this_info{&m_impl};
    do_filter_active(this_info);

    spdlog::info("VertexHalfPlaneTrajectoryFilter PHs: {}.", m_impl.PHs.size());
}

void VertexHalfPlaneTrajectoryFilter::do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info)
{
    FilterTOIInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();
    this_info.m_toi   = info.toi();
    do_filter_toi(this_info);
}


void VertexHalfPlaneTrajectoryFilter::Impl::record_friction_candidates(
    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    loose_resize(friction_PHs, PHs.size());
    friction_PHs.view().copy_from(PHs);
}

void VertexHalfPlaneTrajectoryFilter::do_record_friction_candidates(
    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    m_impl.record_friction_candidates(info);
}

muda::CBufferView<Vector2i> VertexHalfPlaneTrajectoryFilter::PHs() noexcept
{
    return m_impl.PHs;
}

Float VertexHalfPlaneTrajectoryFilter::BaseInfo::d_hat() const noexcept
{
    return m_impl->global_contact_manager->d_hat();
}

Float VertexHalfPlaneTrajectoryFilter::DetectInfo::alpha() const noexcept
{
    return m_alpha;
}

muda::CBufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_normals() const noexcept
{
    return m_impl->half_plane->normals();
}

muda::CBufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_positions() const noexcept
{
    return m_impl->half_plane->positions();
}

muda::CBufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::positions() const noexcept
{
    return m_impl->global_vertex_manager->positions();
}

muda::CBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::surf_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_vertices();
}

muda::CBufferView<Vector3> VertexHalfPlaneTrajectoryFilter::DetectInfo::displacements() const noexcept
{
    return m_impl->global_vertex_manager->displacements();
}

void VertexHalfPlaneTrajectoryFilter::FilterActiveInfo::PHs(muda::CBufferView<Vector2i> PHs) noexcept
{
    m_impl->PHs = PHs;
}

muda::VarView<Float> VertexHalfPlaneTrajectoryFilter::FilterTOIInfo::toi() noexcept
{
    return m_toi;
}
}  // namespace uipc::backend::cuda
