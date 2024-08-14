#pragma once
#include <uipc/geometry/geometry.h>
#include <uipc/geometry/abstract_simplicial_complex.h>
#include <uipc/geometry/simplicial_complex_attributes.h>

namespace uipc::geometry
{
/**
 * @brief A simplicial complex is a collection of simplices.
 * 
 * In $\mathbb{R}^3$, a simplicial complex is defined as follows:
 * $$
 * K = (V, E, F, T),
 * $$
 * where $V$ is the set of vertices, $E$ is the set of edges, $F$ is the set of triangles, and $T$ is the set of tetrahedra.
 * 
 * @sa @ref [Tutorial/Geometry](../tutorial/geometry.md)
 */
class UIPC_CORE_API SimplicialComplex : public Geometry
{
    friend struct fmt::formatter<SimplicialComplex>;

    template <typename T>
    friend class AttributeFriend;

  public:
    /**
     * @brief Alias for the vertex attributes
     * 
     * @sa SimplicialComplexAttributes
     */
    using VertexAttributes  = SimplicialComplexAttributes<VertexSlot>;
    using CVertexAttributes = SimplicialComplexAttributes<const VertexSlot>;
    /**
     * @brief Alias for the edge attributes
     * 
     * @sa SimplicialComplexAttributes
     */
    using EdgeAttributes  = SimplicialComplexAttributes<EdgeSlot>;
    using CEdgeAttributes = SimplicialComplexAttributes<const EdgeSlot>;
    /**
     * @brief Alias for the triangle attributes
     * 
     * @sa SimplicialComplexAttributes
     */
    using TriangleAttributes  = SimplicialComplexAttributes<TriangleSlot>;
    using CTriangleAttributes = SimplicialComplexAttributes<const TriangleSlot>;
    /**
     * @brief Alias for the tetrahedron attributes
     *
     * @sa SimplicialComplexAttributes
     */
    using TetrahedronAttributes = SimplicialComplexAttributes<TetrahedronSlot>;
    using CTetrahedronAttributes = SimplicialComplexAttributes<const TetrahedronSlot>;

    SimplicialComplex();
    SimplicialComplex(const AbstractSimplicialComplex& asc, span<const Vector3> positions);


    SimplicialComplex(const SimplicialComplex& o) = default;
    SimplicialComplex(SimplicialComplex&& o)      = default;

    SimplicialComplex& operator=(const SimplicialComplex& o) = delete;
    SimplicialComplex& operator=(SimplicialComplex&& o)      = delete;

    /**
     * @brief Get the positions of the vertices
     * 
     * @return AttributeSlot<Vector3>& 
     */
    [[nodiscard]] AttributeSlot<Vector3>& positions() noexcept;

    /**
     * @brief A short cut to get the positions of the vertices
     * 
     * @return const AttributeSlot<Vector3>& 
     */
    [[nodiscard]] const AttributeSlot<Vector3>& positions() const noexcept;

    /**
     * @brief A wrapper of the vertices and its attributes of the simplicial complex.
     * 
     * @return VertexAttributeInfo 
     */
    [[nodiscard]] VertexAttributes  vertices() noexcept;
    [[nodiscard]] CVertexAttributes vertices() const noexcept;

    /**
     * @brief A wrapper of the edges and its attributes of the simplicial complex.
     * 
     * @return EdgeAttributes 
     */
    [[nodiscard]] EdgeAttributes  edges() noexcept;
    [[nodiscard]] CEdgeAttributes edges() const noexcept;


    /**
     * @brief  A wrapper of the triangles and its attributes of the simplicial complex.
     * 
     * @return TriangleAttributes 
     */
    [[nodiscard]] TriangleAttributes  triangles() noexcept;
    [[nodiscard]] CTriangleAttributes triangles() const noexcept;
    /**
     * @brief A wrapper of the tetrahedra and its attributes of the simplicial complex.
     * 
     * @return TetrahedronAttributes 
     */
    [[nodiscard]] TetrahedronAttributes  tetrahedra() noexcept;
    [[nodiscard]] CTetrahedronAttributes tetrahedra() const noexcept;

    /**
     * @brief Get the dimension of the simplicial complex.
     *
     * Return the maximum dimension of the simplices in the simplicial complex.
     * 
     * @return IndexT 
     */
    [[nodiscard]] IndexT dim() const noexcept;


  protected:
    virtual std::string_view   get_type() const noexcept override;
    [[nodiscard]] virtual Json do_to_json() const override;

  private:
    AbstractSimplicialComplex m_asc;
    AttributeCollection       m_vertex_attributes;
    AttributeCollection       m_edge_attributes;
    AttributeCollection       m_triangle_attributes;
    AttributeCollection       m_tetrahedron_attributes;
};
}  // namespace uipc::geometry

//formatter

namespace fmt
{
template <>
struct UIPC_CORE_API formatter<uipc::geometry::SimplicialComplex> : formatter<string_view>
{
    appender format(const uipc::geometry::SimplicialComplex& c, format_context& ctx) const;
};
}  // namespace fmt
