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

Timer.enable_all()
Logger.set_level(Logger.Info)
engine = Engine('cuda', './workspace')
world = World(engine)

config = Scene.default_config()
config['dt'] = 1.0/30
config['contact']['d_hat'] = 0.002
config['gravity'] = [[0.0],[-9.8],[0.0]]
config['newton']['velocity_tol'] = 0.05
config['newton']['max_iter'] = 1024
config['linear_system']['tol_rate'] = 1e-3
print(config)
scene = Scene(config)

empty = Empty()
snh = NeoHookeanShell()
dsb = DiscreteShellBending()
spc = SoftPositionConstraint()
io = SimplicialComplexIO()

default_elem = scene.contact_tabular().default_element()

scene.contact_tabular().default_model(0.2, 1e9)
t_shirt_elem = scene.contact_tabular().create('t_shirt')

moduli = ElasticModuli.youngs_poisson(1e7, 0.49)
t_shirt_obj = scene.objects().create('t_shirt')
t_shirt = io.read('/root/libuipc/python/mesh/res_smooth6.obj')
label_surface(t_shirt)
snh.apply_to(t_shirt, moduli=moduli, thickness=0.0005, mass_density=100.0)
t_shirt_elem.apply_to(t_shirt)
t_shirt_obj.geometries().create(t_shirt)

# make body no contact with itself
# body_elem = scene.contact_tabular().create('body')
# scene.contact_tabular().insert(body_elem, body_elem, 0, 0, False)
# scene.contact_tabular().insert(default_elem, body_elem, 0, 0, False)
# scene.contact_tabular().insert(body_elem, t_shirt_elem, 0, 0, False)
# scene.contact_tabular().insert(t_shirt_elem, t_shirt_elem, 0, 0, False)

# io = SimplicialComplexIO()
# girl = io.read('./body_smooth1.obj')
# label_surface(girl)
# empty.apply_to(girl, thickness=0.0)
# spc.apply_to(girl, 1000)
# body_elem.apply_to(girl)
# is_constrained = girl.vertices().find(builtin.is_constrained)
# view(is_constrained)[:] = 1
# is_dynamic = girl.vertices().find(builtin.is_dynamic)
# view(is_dynamic)[:] = 0
# body_gravity = girl.vertices().create(builtin.gravity, Vector3.Zero())
# girl_obj = scene.objects().create('girl')
# slot, rest_slot = girl_obj.geometries().create(girl)

# def update(info: Animation.UpdateInfo):
#     geo_slots:list[GeometrySlot] = info.geo_slots()
#     geo:SimplicialComplex = geo_slots[0].geometry()
#     aim_pos = geo.vertices().find(builtin.aim_position)
#     # aio = AttributeIO (f'./mannequin_obj/frame_{info.frame()}.obj')
#     aio = AttributeIO ('./body_smooth1.obj')
#     aio.read(builtin.position, aim_pos)
#     # view(aim_pos)[:] *= 100

# animator = scene.animator()
# animator.substep(5)
# animator.insert(girl_obj, update)


# ground_obj = scene.objects().create('ground')
# ground_height = 0.0
# g = ground(ground_height)
# ground_obj.geometries().create(g)


world.init(scene)
sio = SceneIO(scene)
sio.write_surface(f"./output/scene_surface{world.frame()}.obj")

while world.frame() < 10:
    world.advance()
    world.retrieve()
#     Timer.report()
    sio.write_surface(f"./output/scene_surface{world.frame()}.obj")
        