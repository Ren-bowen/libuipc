namespace uipc::geometry
{
template <bool IsConst>
void Geometry::InstanceAttributesT<IsConst>::resize(size_t size) &&
    requires(!IsConst)
{
    m_attributes.resize(size);
}

template <bool IsConst>
void Geometry::InstanceAttributesT<IsConst>::reserve(size_t size) &&
    requires(!IsConst)
{
    m_attributes.reserve(size);
}

template <bool IsConst>
void Geometry::InstanceAttributesT<IsConst>::clear() &&
    requires(!IsConst)
{
    m_attributes.clear();
}

template <bool IsConst>
SizeT Geometry::InstanceAttributesT<IsConst>::size() &&
{
    return m_attributes.size();
}

template <bool IsConst>
void Geometry::InstanceAttributesT<IsConst>::destroy(std::string_view name) &&
    requires(!IsConst)
{
    m_attributes.destroy(name);
}
}  // namespace uipc::geometry
