import numpy as np
import polyscope as ps
from polyscope import imgui

from pyuipc_loader import pyuipc
from pyuipc import Vector3, Transform, Logger, Timer
from pyuipc import builtin

from pyuipc import view
from pyuipc.core import *
from pyuipc.geometry import *
from pyuipc.constitution import *
from pyuipc_utils.gui import *

def simulate():
    Timer.enable_all()
    Logger.set_level(Logger.Info)
    engine = Engine('cuda', './workspace')
    world = World(engine)

    p1 = np.array([0.111477, 0.100992, -1.419501])
    p2 = np.array([0.094850, 0.103442, -1.007820])
    dir = (p2 - p1) / np.linalg.norm(p2 - p1) * 9.8
    config = Scene.default_config()
    config['dt'] = 1.0/30
    config['contact']['d_hat'] = 0.001
    config['gravity'] = [[dir[0]],[dir[1]],[dir[2]]]
    config['newton']['velocity_tol'] = 0.05
    config['newton']['max_iter'] = 1024
    config['extras']['debug']['dump_surface'] = False
    config['linear_system']['tol_rate'] = 1e-3
    # config['cfl']['enable'] = True
    print(config)
    scene = Scene(config)

    empty = Empty()
    snh = NeoHookeanShell()
    dsb = DiscreteShellBending()
    spc = SoftPositionConstraint()
    io = SimplicialComplexIO()

    default_elem = scene.contact_tabular().default_element()

    scene.contact_tabular().default_model(1, 1e9)
    t_shirt_elem = scene.contact_tabular().create('t_shirt')

    moduli = ElasticModuli.youngs_poisson(1e5, 0.49)
    t_shirt_obj = scene.objects().create('t_shirt')
    t_shirt = io.read('/root/libuipc/python/mesh/init_mesh_dress4.obj')
    label_surface(t_shirt)
    snh.apply_to(t_shirt, moduli=moduli, thickness=0.0005, mass_density=100.0)
    dsb.apply_to(t_shirt, E=1)
    t_shirt_elem.apply_to(t_shirt)

    rest_t_shirt = t_shirt.copy()
    aio = AttributeIO('/root/libuipc/python/mesh/opt_mesh_dress.obj')
    aio.read(builtin.position, rest_t_shirt.positions())
    geo_slot, _ = t_shirt_obj.geometries().create(t_shirt, rest_t_shirt)

    # make body no contact with itself
    body_elem = scene.contact_tabular().create('body')
    scene.contact_tabular().insert(body_elem, body_elem, 0, 0, False)
    scene.contact_tabular().insert(default_elem, body_elem, 0, 0, False)
    # scene.contact_tabular().insert(body_elem, t_shirt_elem, 0, 0, False)
    # scene.contact_tabular().insert(t_shirt_elem, t_shirt_elem, 0, 0, False)

    io = SimplicialComplexIO()
    
    girl = io.read('/root/libuipc/python/mesh/body_smooth_212.obj')
    label_surface(girl)
    empty.apply_to(girl, thickness=0.0)
    spc.apply_to(girl, 1000)
    body_elem.apply_to(girl)
    is_constrained = girl.vertices().find(builtin.is_constrained)
    view(is_constrained)[:] = 1
    is_dynamic = girl.vertices().find(builtin.is_dynamic)
    view(is_dynamic)[:] = 0
    body_gravity = girl.vertices().create(builtin.gravity, Vector3.Zero())
    girl_obj = scene.objects().create('girl')
    slot, rest_slot = girl_obj.geometries().create(girl)

    def update(info: Animation.UpdateInfo):
        geo_slots:list[GeometrySlot] = info.geo_slots()
        geo:SimplicialComplex = geo_slots[0].geometry()
        aim_pos = geo.vertices().find(builtin.aim_position)
        aio = AttributeIO (f'/root/libuipc/python/mesh/body_mesh/body_{info.frame()}.obj')
        # aio = AttributeIO ('/root/libuipc/python/mesh/body_smooth_212.obj')
        aio.read(builtin.position, aim_pos)
        # view(aim_pos)[:] *= 100

    animator = scene.animator()
    animator.substep(20)
    animator.insert(girl_obj, update)

    # ground_obj = scene.objects().create('ground')
    # ground_height = 0.0
    # g = ground(ground_height)
    # ground_obj.geometries().create(g)


    world.init(scene)
    sio = SceneIO(scene)
    sio.write_surface(f'output/scene_surface{world.frame()}.obj')
    io = SimplicialComplexIO()
    io.write(f'output/cloth_surface{world.frame()}.obj', geo_slot.geometry())

    run = False
    # sgui = SceneGUI(scene)
    # tri_surf, line_surf, points = sgui.register()

    while world.frame() < 100:
        world.advance()
        world.retrieve()
        io = SimplicialComplexIO()
        io.write(f"output/cloth_surface{world.frame()}.obj", geo_slot.geometry())
        sio.write_surface(f"output/scene_surface{world.frame()}.obj")
