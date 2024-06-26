#include <sim_engine.h>
#include <log_pattern_guard.h>
#include <dof_predictor.h>
#include <global_vertex_manager.h>
#include <global_surface_manager.h>
#include <line_search/line_searcher.h>
#include <gradient_hessian_computer.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::cuda
{
void SimEngine::do_advance()
{
    LogGuard guard;

    // The Pipeline

    AABB vertex_bounding_box = m_global_vertex_manager->compute_vertex_bounding_box();
    Float box_size = vertex_bounding_box.diagonal().norm();

    {
        // 1. Predict Motion => x_tilde = x + v * dt
        m_state = SimEngineState::PredictMotion;
        m_dof_predictor->predict();

        // 2. Build Collision Pairs

        // 3. Select Adaptive Kappa

        // 4. Nonlinear-Newton Iteration
        Float tol  = std::min(m_newton_tol * box_size, m_abs_tol);
        SizeT iter = 0;
        while(iter++ < m_newton_max_iter)
        {
            // 1) Build Collision Pairs

            m_state = SimEngineState::ComputeGradientHassian;
            // 2) Compute Contact Gradient and Hessian => G:Vector3, H:Matrix3x3


            // 3) Compute System Gradient and Hessian => G:Vector3, H:Matrix3x3
            // E.g. FEM/ABD ...
            m_gradient_hessian_computer->compute_gradient_hessian();


            m_state = SimEngineState::SolveGlobalLinearSystem;
            // 4) Solve Global Linear System => dx = A^-1 * b
            m_global_linear_system->solve();

            // 5) Get Max Movement => dx_max = max(|dx|), if dx_max < tol, break
            Float res = m_global_vertex_manager->compute_max_displacement();
            spdlog::info("Newton Iteration: {} Residual: {}/{}", iter, res, tol);
            if(res < tol)
                break;

            m_state = SimEngineState::LineSearch;
            // 6) Begin Line Search
            {
                // Record Current State x to x_0
                m_line_searcher->record_start_point();

                // Compute Current Energy => E_0
                Float E0 = m_line_searcher->compute_energy();

                auto compute_energy = [&](Float alpha) -> Float
                {
                    // Step Forward => x = x_0 + alpha * dx
                    m_line_searcher->step_forward(alpha);
                    // Compute New Energy => E
                    return m_line_searcher->compute_energy();
                };

                // Build Continuous Collision Pairs with alpha = 1.0
                Float alpha = 1.0;

                SizeT max_line_search_iter = 1000;  // now just hard code it
                SizeT line_search_iter     = 0;
                while(line_search_iter++ < max_line_search_iter)  // Energy Test
                {
                    Float E               = compute_energy(alpha);
                    bool  energy_decrease = E < E0;  // Check Energy Decrease
                    bool  no_inversion    = true;    // Check Inversion

                    bool success = energy_decrease && no_inversion;
                    if(success)
                        break;

                    // If not success, then shrink alpha
                    alpha /= 2;
                    E = compute_energy(alpha);
                    //spdlog::info("Line Search Iteration: {} Alpha: {} Energy: {}/{}",
                    //             line_search_iter,
                    //             alpha,
                    //             E,
                    //             E0);
                }
            }
        }

        // 5. Update Velocity => v = (x - x_0) / dt
        m_state = SimEngineState::UpdateVelocity;
        m_dof_predictor->compute_velocity();
    }

    // Trigger the rebuild_scene event, systems register their actions will be called here
    m_state = SimEngineState::RebuildScene;
    {
        event_rebuild_scene();

        // TODO: rebuild the vertex and surface info
        // m_global_vertex_manager->rebuild_vertex_info();
        // m_global_surface_manager->rebuild_surface_info();
    }

    // After the rebuild_scene event, the pending creation or deletion can be solved
    auto scene = m_world_visitor->scene();
    scene.solve_pending();
}
}  // namespace uipc::backend::cuda
