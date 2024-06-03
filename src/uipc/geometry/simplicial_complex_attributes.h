#pragma once
#include <uipc/geometry/simplex_slot.h>
#include <uipc/geometry/attribute_collection.h>

namespace uipc::geometry
{
template <std::derived_from<ISimplexSlot> SimplexSlotT>
class SimplicialComplexAttributes;

template <std::derived_from<ISimplexSlot> SimplexSlotT>
class SimplicialComplexTopo;

/**
* @brief A wrapper of the topology of the simplicial complex.
*/
template <std::derived_from<ISimplexSlot> SimplexSlotT>
class SimplicialComplexTopo
{
    friend struct fmt::formatter<SimplicialComplexTopo<SimplexSlotT>>;
    friend class SimplicialComplexAttributes<SimplexSlotT>;

    // Note: SimplexSlotT can be const or non-const
    using NonConstSimplexSlotT = std::remove_const_t<SimplexSlotT>;
    using ConstSimplexSlotT    = std::add_const_t<NonConstSimplexSlotT>;

    friend class SimplicialComplexTopo<NonConstSimplexSlotT>;

  public:
    /**
     * @brief Get a non-const view of the topology, this function may clone the data.
     */
    template <IndexT N>
    friend span<typename SimplexSlot<N>::ValueT> view(SimplicialComplexTopo<SimplexSlot<N>>&& v);

    /**
     * @brief Get the backend view of the topology, this function guarantees no data clone.
     */
    template <IndexT N>
    friend backend::BufferView backend_view(SimplicialComplexTopo<SimplexSlot<N>>&& v) noexcept;

    template <IndexT N>
    friend backend::BufferView backend_view(SimplicialComplexTopo<const SimplexSlot<N>>&& v) noexcept;

    /**
     * @brief Get a const view of the topology, this function guarantees no data clone.
     */
    [[nodiscard]] auto view() && noexcept { return m_topology.view(); }
    /**
     * @brief Query if the topology is owned by current simplicial complex.
     */
    [[nodiscard]] bool is_shared() && noexcept;

    void share(SimplicialComplexTopo<ConstSimplexSlotT>&& topo) && noexcept;

    operator SimplicialComplexTopo<ConstSimplexSlotT>() && noexcept
    {
        return SimplicialComplexTopo<ConstSimplexSlotT>(m_topology);
    }

  private:
    SimplicialComplexTopo(SimplexSlotT& topo);
    SimplexSlotT& m_topology;
};

template <>
class SimplicialComplexTopo<const VertexSlot>
{
    friend struct fmt::formatter<SimplicialComplexTopo<const VertexSlot>>;
    friend class SimplicialComplexAttributes<const VertexSlot>;
    friend class SimplicialComplexTopo<VertexSlot>;

  public:
    /**
     * @brief Get the backend view of the topology, this function guarantees no data clone.
     */
    friend backend::BufferView backend_view(SimplicialComplexTopo<const VertexSlot>&& v) noexcept;

    /**
     * @brief Get a const view of the topology, this function guarantees no data clone.
     */
    [[nodiscard]] auto view() && noexcept { return m_topology.view(); }

    /**
     * @brief Query if the topology is owned by current simplicial complex.
     */
    [[nodiscard]] bool is_shared() && noexcept;

  private:
    SimplicialComplexTopo(const VertexSlot& topo);
    const VertexSlot& m_topology;
};

template <>
class SimplicialComplexTopo<VertexSlot>
{
    friend struct fmt::formatter<SimplicialComplexTopo<VertexSlot>>;
    friend class SimplicialComplexAttributes<VertexSlot>;

  public:
    /**
     * @brief Get a non-const view of the topology, this function may clone the data.
     */
    friend span<typename VertexSlot::ValueT> view(SimplicialComplexTopo<VertexSlot>&& v);

    /**
     * @brief Get the backend view of the topology, this function guarantees no data clone.
     */
    friend backend::BufferView backend_view(SimplicialComplexTopo<VertexSlot>&& v) noexcept;

    /**
     * @brief Get a const view of the topology, this function guarantees no data clone.
     */
    [[nodiscard]] auto view() && noexcept { return m_topology.view(); }
    /**
     * @brief Query if the topology is owned by current simplicial complex.
     */
    [[nodiscard]] bool is_shared() && noexcept;

    void share(SimplicialComplexTopo<const VertexSlot>&& topo) && noexcept;

    operator SimplicialComplexTopo<const VertexSlot>() && noexcept
    {
        return SimplicialComplexTopo<const VertexSlot>(m_topology);
    }

  private:
    SimplicialComplexTopo(VertexSlot& topo);
    VertexSlot& m_topology;
};


/**
 * @brief A collection of attributes for a specific type of simplices. The main API for accessing the attributes of a simplicial complex.
 */
template <std::derived_from<ISimplexSlot> SimplexSlotT>
class SimplicialComplexAttributes
{
    friend struct fmt::formatter<SimplicialComplexAttributes<SimplexSlotT>>;

    using Topo = SimplicialComplexTopo<SimplexSlotT>;
    using AutoAttributeCollection =
        std::conditional_t<std::is_const_v<SimplexSlotT>, const AttributeCollection, AttributeCollection>;

  public:
    SimplicialComplexAttributes(const SimplicialComplexAttributes& o) = default;
    SimplicialComplexAttributes(SimplicialComplexAttributes&& o)      = default;
    SimplicialComplexAttributes& operator=(const SimplicialComplexAttributes& o) = default;
    SimplicialComplexAttributes& operator=(SimplicialComplexAttributes&& o) = default;


    /**
	 * @brief Get the topology of the simplicial complex.
	 * 
	 * @return Topo 
	 */
    [[nodiscard]] Topo topo() noexcept;

    [[nodiscard]] Topo topo() const noexcept;

    /**
     * @sa [AttributeCollection::resize()](../AttributeCollection/#resize)
     */
    void resize(SizeT size)
        requires(!std::is_const_v<SimplexSlotT>);
    /**
     * @sa [AttributeCollection::reserve()](../AttributeCollection/#reserve)
     */
    void reserve(SizeT size)
        requires(!std::is_const_v<SimplexSlotT>);
    /**
     * @sa [AttributeCollection::clear()](../AttributeCollection/#clear)
     */
    void clear()
        requires(!std::is_const_v<SimplexSlotT>);
    /**
     * @sa [AttributeCollection::size()](../AttributeCollection/#size)
     */
    [[nodiscard]] SizeT size() const noexcept;
    /**
     * @sa [AttributeCollection::destroy()](../AttributeCollection/#destroy) 
     */
    void destroy(std::string_view name)
        requires(!std::is_const_v<SimplexSlotT>);

    /**
     * @brief Find an attribute by type and name, if the attribute does not exist, return nullptr.
     */
    template <typename T>
    [[nodiscard]] decltype(auto) find(std::string_view name)
    {
        return m_attributes.template find<T>(name);
    }

    /**
    * @brief Find an attribute by type and name, if the attribute does not exist, return nullptr.
    */
    template <typename T>
    [[nodiscard]] decltype(auto) find(std::string_view name) const
    {
        return std::as_const(m_attributes).template find<T>(name);
    }

    template <typename T>
    decltype(auto) create(std::string_view name, const T& default_value = {})
    {
        return m_attributes.template create<T>(name, default_value);
    }

    template <typename T>
    decltype(auto) share(std::string_view name, const AttributeSlot<T>& slot)
    {
        return m_attributes.template share<T>(name, slot);
    }

  private:
    friend class SimplicialComplex;
    SimplexSlotT&            m_topology;
    AutoAttributeCollection& m_attributes;

    SimplicialComplexAttributes(SimplexSlotT&            topology,
                                AutoAttributeCollection& attributes) noexcept;
};
}  // namespace uipc::geometry


namespace fmt
{
template <std::derived_from<uipc::geometry::ISimplexSlot> SimplexSlotT>
struct formatter<uipc::geometry::SimplicialComplexTopo<SimplexSlotT>>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const uipc::geometry::SimplicialComplexTopo<SimplexSlotT>& topo,
                FormatContext&                                             ctx)
    {
        uipc::geometry::ISimplexSlot& i_topo = topo.m_topology;
        return fmt::format_to(
            ctx.out(), "size={}, is_shared={}", i_topo.size(), i_topo.is_shared());
    }
};

template <std::derived_from<uipc::geometry::ISimplexSlot> SimplexSlotT>
struct formatter<uipc::geometry::SimplicialComplexAttributes<SimplexSlotT>>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const uipc::geometry::SimplicialComplexAttributes<SimplexSlotT>& attributes,
                FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "{}", attributes.m_attributes);
    }
};
}  // namespace fmt


#include "details/simplicial_complex_attributes.inl"
