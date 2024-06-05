#include <catch.hpp>
#include <uipc/world/scene.h>
#include <uipc/common/enumerate.h>
#include <app/asset_dir.h>
#include <uipc/geometry.h>
#include <iostream>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitutions/affine_body.h>
#include <uipc/common/format.h>
#include <uipc/world/object.h>

using namespace uipc;
using namespace uipc::world;
using namespace uipc::constitution;

TEST_CASE("abd", "[constitution]")
{
    Scene scene;
    auto& constitution_tabular = scene.constitution_tabular();
    auto& abd = constitution_tabular.create<AffineBodyConstitution>();

    geometry::SimplicialComplexIO io;
    auto mesh0 = io.read_msh(fmt::format("{}cube.msh", AssetDir::tetmesh_path()));
    mesh0.instances().resize(5);

    auto wood_abd = abd.create_material(1e8);
    auto kappa = wood_abd.apply_to(mesh0);
    
    REQUIRE(kappa->size() == mesh0.instances().size());
    REQUIRE(std::ranges::all_of(kappa->view(), [](auto v) { return v == 1e8; }));

    // this name is important for the AffineBodyConstitution
    REQUIRE(kappa->name() == "abd::kappa");
}