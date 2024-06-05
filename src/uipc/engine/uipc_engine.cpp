#include <uipc/engine/uipc_engine.h>
#include <dylib.hpp>

namespace uipc::engine
{
class UIPCEngine::Impl
{
    std::string m_backend_name;
    dylib       m_module;
    using Deleter      = void (*)(IEngine*);
    IEngine* m_engine  = nullptr;
    Deleter  m_deleter = nullptr;

  public:
    Impl(std::string_view backend_name)
        : m_backend_name(backend_name)
        , m_module{fmt::format("uipc_backend_{}", backend_name)}
    {
        auto creator = m_module.get_function<IEngine*()>("uipc_create_engine");
        if(!creator)
            throw EngineException{fmt::format("Can't find backend [{}]'s engine creator.",
                                              backend_name)};
        m_engine = creator();
        m_deleter = m_module.get_function<void(IEngine*)>("uipc_destroy_engine");
        if(!m_deleter)
            throw EngineException{fmt::format("Can't find backend [{}]'s engine deleter.",
                                              backend_name)};
    }

    std::string_view backend_name() const { return m_backend_name; }

    void init(backend::WorldVisitor v) { m_engine->init(v); }

    void advance() { m_engine->advance(); }

    void sync() { m_engine->sync(); }

    void retrieve() { m_engine->retrieve(); }

    ~Impl()
    {
        UIPC_ASSERT(m_deleter && m_engine, "Engine not initialized, why can it happen?");
        m_deleter(m_engine);
    }
};


UIPCEngine::UIPCEngine(std::string_view backend_name)
    : m_impl{std::make_unique<Impl>(backend_name)}
{
}

UIPCEngine::~UIPCEngine() {}

void UIPCEngine::do_init(backend::WorldVisitor v)
{
    m_impl->init(v);
}

void UIPCEngine::do_advance()
{
    m_impl->advance();
}

void UIPCEngine::do_sync()
{
    m_impl->sync();
}

void UIPCEngine::do_retrieve()
{
    m_impl->retrieve();
}
}  // namespace uipc::engine