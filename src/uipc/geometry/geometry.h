#pragma once
#include <string_view>
#include <uipc/geometry/attribute_collection.h>

namespace uipc::geometry
{
/**
 * @brief An abstract class for geometries
 */
class IGeometry
{
  public:
    /**
     * @brief Get the type of the geometries, check the type to downcast the geometries to a specific type
     * 
     * @return a string_view of the type of the geometries
     */
    [[nodiscard]] std::string_view type() const noexcept;
    virtual ~IGeometry() = default;

  protected:
    [[nodiscard]] virtual std::string_view get_type() const noexcept = 0;
};

/**
 * @brief A base geometries class that contains the instance attributes and the meta attributes.
 */
class Geometry : public IGeometry
{
  public:
    /**
     * @brief A wrapper class for the meta attributes of a geometries.
     */
    template <bool IsConst>
    class MetaAttributesT
    {
        friend struct fmt::formatter<MetaAttributesT<IsConst>>;

        using AutoAttributeCollection =
            std::conditional_t<IsConst, const AttributeCollection, AttributeCollection>;

      public:
        MetaAttributesT(AutoAttributeCollection& attributes)
            : m_attributes(attributes)
        {
        }

        MetaAttributesT(const MetaAttributesT& o)            = default;
        MetaAttributesT(MetaAttributesT&& o)                 = default;
        MetaAttributesT& operator=(const MetaAttributesT& o) = default;
        MetaAttributesT& operator=(MetaAttributesT&& o)      = default;

        /**
         * @brief Find an attribute by type and name, if the attribute does not exist, return nullptr.
         */
        template <typename T>
        [[nodiscard]] auto find(std::string_view name) &&
        {
            return m_attributes.template find<T>(name);
        }

        /**
         * @brief Create an attribute with the given name.
         */
        template <typename T>
        decltype(auto) create(std::string_view name, const T& init_value = {}) &&
        {
            return m_attributes.template create<T>(name, init_value);
        }

      private:
        AutoAttributeCollection& m_attributes;
    };

    using MetaAttributes  = MetaAttributesT<false>;
    using CMetaAttributes = MetaAttributesT<true>;

    /**
     * @brief A wrapper class for the instance attributes of a geometries.
     */
    template <bool IsConst>
    class InstanceAttributesT
    {
        friend struct fmt::formatter<InstanceAttributesT<IsConst>>;

        using AutoAttributeCollection =
            std::conditional_t<IsConst, const AttributeCollection, AttributeCollection>;

      public:
        InstanceAttributesT(AutoAttributeCollection& attributes)
            : m_attributes(attributes)
        {
        }
        InstanceAttributesT(const InstanceAttributesT& o)            = default;
        InstanceAttributesT(InstanceAttributesT&& o)                 = default;
        InstanceAttributesT& operator=(const InstanceAttributesT& o) = default;
        InstanceAttributesT& operator=(InstanceAttributesT&& o)      = default;

        /**
         * @sa [AttributeCollection::resize()](../../AttributeCollection/#resize)
         */
        void resize(size_t size) &&
            requires(!IsConst);
        /**
         * @sa [AttributeCollection::reserve()](../../AttributeCollection/#reserve)
         */
        void reserve(size_t size) &&
            requires(!IsConst);
        /**
         * @sa [AttributeCollection::clear()](../../AttributeCollection/#clear)
         */
        void clear() &&
            requires(!IsConst);
        /**
         * @sa [AttributeCollection::size()](../../AttributeCollection/#size)
         */
        [[nodiscard]] SizeT size() &&;

        /**
         * @sa [AttributeCollection::destroy()](../../AttributeCollection/#destroy) 
         */
        void destroy(std::string_view name) &&
            requires(!IsConst);

        /**
         * @brief Find an attribute by type and name, if the attribute does not exist, return empty OptionalRef.
         */
        template <typename T>
        [[nodiscard]] auto find(std::string_view name) &&
        {
            return m_attributes.template find<T>(name);
        }

        /**
         * @brief Create an attribute with the given name.
         */
        template <typename T>
        decltype(auto) create(std::string_view name, const T& init_value = {}) &&
        {
            return m_attributes.template create<T>(name, init_value);
        }

      private:
        AutoAttributeCollection& m_attributes;
    };

    using InstanceAttributes  = InstanceAttributesT<false>;
    using CInstanceAttributes = InstanceAttributesT<true>;

    Geometry();

    // allow copy and move on construction, because the geometry truely empty
    Geometry(const Geometry& o) = default;
    Geometry(Geometry&& o)      = default;

    // no copy or move assignment, because the geometry is no longer empty
    Geometry& operator=(const Geometry& o) = delete;
    Geometry& operator=(Geometry&& o)      = delete;

    /**
     * @brief A short-cut to get the non-const transforms attribute slot.
     * 
     * @return The attribute slot of the non-const transforms.
     */
    [[nodiscard]] AttributeSlot<Matrix4x4>& transforms();
    /**
     * @brief A short-cut to get the const transforms attribute slot.
     * 
     * @return The attribute slot of the const transforms.
     */
    [[nodiscard]] const AttributeSlot<Matrix4x4>& transforms() const;

    /**
     * @brief Get the meta attributes of the geometries.
     * 
     * @return The meta attributes of the geometries. 
     */
    [[nodiscard]] MetaAttributes meta();

    [[nodiscard]] CMetaAttributes meta() const;


    /**
     * @brief Get the instance attributes of the geometries.
     * 
     * @return  The instance attributes of the geometries.
     */
    [[nodiscard]] InstanceAttributes instances();

    [[nodiscard]] CInstanceAttributes instances() const;

  protected:
    AttributeCollection m_intances;
    AttributeCollection m_meta;
};
}  // namespace uipc::geometry


namespace fmt
{
template <bool IsConst>
struct formatter<uipc::geometry::Geometry::MetaAttributesT<IsConst>>
    : public formatter<string_view>
{
    auto format(const uipc::geometry::Geometry::MetaAttributesT<IsConst>& attr,
                format_context&                                           ctx)
    {
        return fmt::format_to(ctx.out(), "{}", attr.m_attributes);
    }
};

template <bool IsConst>
struct formatter<uipc::geometry::Geometry::InstanceAttributesT<IsConst>>
    : public formatter<string_view>
{
    auto format(const uipc::geometry::Geometry::InstanceAttributesT<IsConst>& attr,
                format_context& ctx)
    {
        return fmt::format_to(ctx.out(), "{}", attr.m_attributes);
    }
};

template <>
struct formatter<uipc::geometry::Geometry> : public formatter<string_view>
{
    auto format(const uipc::geometry::Geometry& geo, format_context& ctx)
    {
        return fmt::format_to(ctx.out(),
                              R"(type:<{}>;
meta:{};
instances:{};)",
                              geo.type(),
                              geo.meta(),
                              geo.instances());
    }
};
}  // namespace fmt
#include "details/geometry.inl"