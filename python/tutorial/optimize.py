import os
import torch
import numpy as np
import trimesh
from clothed_human import simulate
from smpl_util import JointMapper, smpl_to_openpose
from rotation_utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
import igl
import smplx

def piont_edge_distance(p, v1, v2):
    t = torch.sum((p - v1) * (v2 - v1), dim=-1) / torch.sum((v2 - v1) ** 2, dim=-1)
    result = torch.where(
        t < 0,
        torch.norm(p - v1, dim=-1),
        torch.where(
            t > 1,
            torch.norm(p - v2, dim=-1),
            torch.norm(p - (v1 + t.unsqueeze(-1) * (v2 - v1)), dim=-1)
        )
    )
    return result


def distance_loss(eps, p, v, e):
    # p = torch.tensor(p, dtype=torch.float32, device='cuda')
    v = torch.tensor(v, dtype=torch.float32, device='cuda')
    e = torch.tensor(e, dtype=torch.int64, device='cuda')
    v1 = v[e[:, 0]]
    v2 = v[e[:, 1]]
    v3 = v[e[:, 2]]
    # center = (v1 + v2 + v3) / 3  
    # min_idx = torch.argmin(torch.norm(p.unsqueeze(1) - center.unsqueeze(0), dim=-1), dim=1)
    _, min_idx, _ = igl.point_mesh_squared_distance(p.detach().cpu().numpy(), v.detach().cpu().numpy(), e.detach().cpu().numpy())
    # normals = torch.cross(v2[min_idx] - v1[min_idx], v3[min_idx] - v1[min_idx], dim=-1)
    v1 = v1[min_idx, :]
    v2 = v2[min_idx, :]
    v3 = v3[min_idx, :]
    # edge1 = v2 - v1
    #  edge2 = v3 - v1
    # normal = torch.cross(edge1, edge2, dim=-1)
    # normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-10)
    # distance = torch.sum((p - v1) * normal, dim=-1)
    # proj_p = p - distance.unsqueeze(1) * normal
    # proj_p = t0 * v1 + t1 * v2 + t2 * v3
    LHS = torch.stack([
        torch.stack([
            torch.sum(v1 * (v2 - v1), dim=1),
            torch.sum(v2 * (v2 - v1), dim=1),
            torch.sum(v3 * (v2 - v1), dim=1)
        ], dim=1),
        torch.stack([
            torch.sum(v1 * (v3 - v1), dim=1),
            torch.sum(v2 * (v3 - v1), dim=1),
            torch.sum(v3 * (v3 - v1), dim=1)
        ], dim=1),
        torch.stack([
            torch.tensor(1, device='cuda').expand(v1.size(0)),
            torch.tensor(1, device='cuda').expand(v1.size(0)),
            torch.tensor(1, device='cuda').expand(v1.size(0))
        ], dim=1)
    ], dim=2)

    Rhs = torch.stack([
        torch.sum(p * (v2 - v1), dim=1),
        torch.sum(p * (v3 - v1), dim=1),
        torch.ones(p.size(0), device='cuda')
    ], dim=1)
    t = torch.linalg.solve(LHS, Rhs)
    distance = torch.norm(p - t[0] * v1 + t[1] * v2 + t[2] * v3, dim=-1)
    distance1 = piont_edge_distance(p, v[e[min_idx, 0]], v[e[min_idx, 1]])
    distance2 = piont_edge_distance(p, v[e[min_idx, 1]], v[e[min_idx, 2]])
    distance3 = piont_edge_distance(p, v[e[min_idx, 2]], v[e[min_idx, 0]])
    # if t[i][0] < 0 or t[i][1] < 0 or t[i][2] < 0:
    #     distance[i] = torch.min(torch.min(distance1, distance2), distance3)
    distance = torch.where(
        torch.logical_or(
            torch.logical_or(t[:, 0] < 0, t[:, 1] < 0),
            t[:, 2] < 0
        ),
        torch.min(torch.min(distance1, distance2), distance3),
        distance
    )
    loss = torch.sum(torch.max(eps - distance, torch.tensor(0.0, device=p.device)))
    return loss

def optimize(v_init, v_k, v_goal, f, iter):
    # X_init (=v_init[f]), X_k (=v_k[f]) : numpy array (N, 3, 3), 3d mesh vertices
    v_init = torch.tensor(v_init, dtype=torch.float32, device='cuda')
    v_k = torch.tensor(v_k, dtype=torch.float32, device='cuda')
    v_goal = torch.tensor(v_goal, dtype=torch.float32, device='cuda')
    f = torch.tensor(f, dtype=torch.int64, device='cuda')
    X_init = v_init[f]
    X_k = v_k[f]
    X_goal = v_goal[f]
    X_ig = torch.zeros_like(X_k)
    # cot[:][0]: angle opposite u, cot[:][2]: angle opposite v, cot[:][1]: the least angle
    cot = torch.ones_like(f)
    # transfer cot from int64 to float32
    cot = cot.float()
    loss = torch.tensor([0], dtype=torch.float32, device='cuda')
    for i in range(0, X_init.size(0)):
        u_init = X_init[i][1] - X_init[i][0]
        v_init = X_init[i][2] - X_init[i][0]
        cos_init = torch.dot(u_init, v_init) / (torch.norm(u_init) * torch.norm(v_init))
        sin_init = torch.sqrt(1 - cos_init ** 2)
        u_goal = X_goal[i][1] - X_goal[i][0]
        v_goal = X_goal[i][2] - X_goal[i][0]
        cos_goal = torch.dot(u_goal, v_goal) / (torch.norm(u_goal) * torch.norm(v_goal))
        sin_goal = torch.sqrt(1 - cos_goal ** 2)
        u = X_k[i][1] - X_k[i][0]
        v = X_k[i][2] - X_k[i][0]
        cos = torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
        sin = torch.sqrt(1 - cos ** 2)
        
        # transfer to 2d coordinate
        T_init = torch.tensor([[torch.norm(u_init), cos_init * torch.norm(v_init)],[0, sin_init * torch.norm(v_init)]], device='cuda')
        T_goal = torch.tensor([[torch.norm(u_goal), cos_goal * torch.norm(v_goal)],[0, sin_goal * torch.norm(v_goal)]], device='cuda')
        T = torch.tensor([[torch.norm(u), cos * torch.norm(v)],[0, sin * torch.norm(v)]], device='cuda')

        # calculate the process matrix 
        # T = A @ T_init ,T_goal = A @ T_init_goal -> T_init_goal = A_inv @ T_goal
        # check T_init is invertible
        if torch.abs(sin_init) < 1e-4:
            A = torch.eye(2, device='cuda')
        else:
            A = torch.matmul(T, torch.inverse(T_init))
        if torch.abs(sin) < 1e-4:
            B = torch.eye(2, device='cuda')
        else:
            B = torch.matmul(T_goal, torch.inverse(T))
        # compute the eigenvalues of B
        eigen_values, _ = torch.linalg.eig(B)
        # print("eigen_values.imag = ", eigen_values.imag)
        eigen_values = eigen_values.real
        loss += torch.sqrt((eigen_values[0] - 1) ** 2 + (eigen_values[1] - 1) ** 2)
        T_ig = torch.matmul(torch.inverse(A), T_goal)
        # print("torch.det(A) = ", torch.det(A))
        # print("A = ", A)
        # since T_init and T are both upper triangular matrix, A and T_ig are also upper triangular matrix

        # transfer back to 3d coordinate
        u_ig = torch.tensor([T_ig[0][0], 0, 0], device='cuda')
        v_ig = torch.tensor([T_ig[0][1], T_ig[1][1], 0], device='cuda')
        w_ig = u_ig - v_ig
        
        '''
        u_ig_norm = torch.min(torch.norm(u_goal) * torch.norm(u_init) / torch.norm(u), torch.norm(u_init))
        v_ig_norm = torch.min(torch.norm(v_goal) * torch.norm(v_init) / torch.norm(v), torch.norm(v_init))
        l_ig_norm = torch.min(torch.norm(u_goal - v_goal) * torch.norm(u_init - v_init) / torch.norm(u - v), torch.norm(u_init - v_init))
        
        cos = (u_ig_norm * u_ig_norm + v_ig_norm * v_ig_norm - l_ig_norm * l_ig_norm) / (2 * u_ig_norm * v_ig_norm)
        if (torch.abs(cos) >= 1):
            X_ig[i][0] = torch.tensor([0, 0, 0], device='cuda')
            X_ig[i][1] = X_ig[i][0] + u_init
            X_ig[i][2] = X_ig[i][0] + v_init
            continue
        sin = torch.sqrt(1 - cos ** 2)
        
        u_ig = torch.tensor([u_ig_norm, 0, 0], device='cuda')
        v_ig = torch.tensor([v_ig_norm * cos, v_ig_norm * sin, 0], device='cuda')
        '''
        # local optimization
        # optimize the rotation matrix L by minimizing:
        # cot[0] * ||u - L @ u_ig||^2 + cot[2] * ||v - L @ v_ig||^2 + cot[1] * ||(u - v) - (L @ u_ig - L @ v_ig)||^2
        # since L is a rotation matrix, we have L @ L^T = I, then the problem is equal to that:
        # minimize -2 * (cot[0] * u_ig.T @ L @ u + cot[2] * v_ig.T @ L @ v + cot[1] * (u_ig - v_ig).T @ L @ (u - v)) <=>
        # minimize -L :: X, where X = cot[0] * u @ u_ig.T + cot[2] * v @ v_ig.T + cot[1] * (u - v) @ (u_ig - v_ig).T <=>
        # minimize ||X - L||^2, where ||.|| is the Frobenius norm.
        # This can be solved by Procrustes analysis.
        # Do SVD on X = U @ S @ V.T, then L = U @ V.T
        X = cot[i][0] * torch.outer(u_init, u_ig) + cot[i][2] * torch.outer(v_init, v_ig) + cot[i][1] * torch.outer(u_init - v_init, w_ig)
        U, S, V = torch.svd(X)
        L = torch.matmul(U, V.T)
        # print("-------------")
        # print("torch.norm(u - u_ig)**2 + torch.norm(v - v_ig)**2 + torch.norm((u - v) - (u_ig - v_ig))**2 = ", torch.norm(u - u_ig)**2 + torch.norm(v - v_ig)**2 + torch.norm((u - v) - (u_ig - v_ig))**2)
        u_ig = torch.matmul(L, u_ig)
        v_ig = torch.matmul(L, v_ig)
        # print("torch.norm(u - u_ig)**2 + torch.norm(v - v_ig)**2 + torch.norm((u - v) - (u_ig - v_ig))**2 = ", torch.norm(u - u_ig)**2 + torch.norm(v - v_ig)**2 + torch.norm((u - v) - (u_ig - v_ig))**2)
        # print("-------------")
        X_ig[i][0] = torch.tensor([0, 0, 0], device='cuda')
        X_ig[i][1] = X_ig[i][0] + u_ig
        X_ig[i][2] = X_ig[i][0] + v_ig

        # print("i = ", i)

    loss = loss / X_init.size(0)
    edges_init = torch.norm(X_init[:, 0] - X_init[:, 1], dim=1) + torch.norm(X_init[:, 1] - X_init[:, 2], dim=1) + torch.norm(X_init[:, 2] - X_init[:, 0], dim=1)
    edges_k = torch.norm(X_k[:, 0] - X_k[:, 1], dim=1) + torch.norm(X_k[:, 1] - X_k[:, 2], dim=1) + torch.norm(X_k[:, 2] - X_k[:, 0], dim=1)
    edges_goal = torch.norm(X_goal[:, 0] - X_goal[:, 1], dim=1) + torch.norm(X_goal[:, 1] - X_goal[:, 2], dim=1) + torch.norm(X_goal[:, 2] - X_goal[:, 0], dim=1)
    edges_ig = torch.norm(X_ig[:, 0] - X_ig[:, 1], dim=1) + torch.norm(X_ig[:, 1] - X_ig[:, 2], dim=1) + torch.norm(X_ig[:, 2] - X_ig[:, 0], dim=1)
    length = torch.tensor([torch.sum(edges_init), torch.sum(edges_k), torch.sum(edges_goal), torch.sum(edges_ig)], device='cuda')
    length = length.cpu().numpy()

    # v_opt = torch.zeros_like(v_k)
    # global optimization : optimize v_opt by minimizing the energy function
    # E(u, L) = \sum_{t=1}^T \sum_{i=0}^2 \left\| \left( u_t^i - u_t^{i+1} \right) 
    # - L_t \left( x_t^i - x_t^{i+1} \right) \right\|^2
    # dEdut^0 = 2 \sum_{t=1}^T \left( u_t^0 - u_t^1 - L_t \left( x_t^0 - x_t^1 \right) \right) \cdot \left( x_t^0 - x_t^1 \right)
    # dEdut^1 = 2 \sum_{t=1}^T \left( u_t^1 - u_t^2 - L_t \left( x_t^1 - x_t^2 \right) \right) \cdot \left( x_t^1 - x_t^2 \right)
    # dEdut^2 = 2 \sum_{t=1}^T \left( u_t^2 - u_t^0 - L_t \left( x_t^2 - x_t^0 \right) \right) \cdot \left( x_t^2 - x_t^0 \right)
    # Unconstrained Quadratic Optimization Problem:
    # v Q v^T + p v^T + c
    # for i in range(f.size(0)):
    #    for j in range(f.size(1)):
    #       Q[f[i][j]][f[i][j]] += 2
    #       Q[f[i][j]][f[i][(j+1)%3]] += -1
    #       Q[f[i][j]][f[i][(j+2)%3]] += -1 
    #       p[f[i][j]] -= 4*X_ig[i][j] - 2*X_ig[i][(j+1)%3] - 2*X_ig[i][(j+2)%3]
    n = v_k.shape[0]
    Q_sparse = sp.lil_matrix((n * 3, n * 3)) 
    p = np.zeros(n * 3)
    f = f.cpu().numpy()
    assert torch.isnan(X_ig).any() == False
    X_ig = X_ig.cpu().numpy()

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(3):
                assert (cot[i][j] + cot[i][(j + 2) % 3]) > 1e-6
                Q_sparse[3 * f[i][j] + k, 3 * f[i][j] + k] += cot[i][j] + cot[i][(j + 2) % 3]
                Q_sparse[3 * f[i][j] + k, 3 * f[i][(j + 1) % 3] + k] += -cot[i][j]
                Q_sparse[3 * f[i][j] + k, 3 * f[i][(j + 2) % 3] + k] += -cot[i][(j + 2) % 3]
                p[3 * f[i][j] + k] += cot[i][j] * (X_ig[i][j][k] - X_ig[i][(j + 1) % 3][k]) + cot[i][(j + 2) % 3] * (X_ig[i][j][k] - X_ig[i][(j + 2) % 3][k])

    Q_sparse = Q_sparse.tocsr()
    v_opt = spla.spsolve(Q_sparse, p).reshape(-1, 3)
    return v_opt, loss

def optimize_body(v_k, v_init, f, body_verts, body_faces, smpl_model, beta=None, pose=None, orient=None, transl=None):
    body_verts_init = body_verts
    body_verts_init = torch.tensor(body_verts_init, dtype=torch.float32, device='cuda')
    d_hat = 3e-3
    dist_init, _, _ = igl.point_mesh_squared_distance(body_verts, v_init, f)
    dist_k,_,_ = igl.point_mesh_squared_distance(body_verts, v_k, f)
    mask = dist_k < (d_hat / 25)
    dist_init = np.sqrt(dist_init)
    dist_k = np.sqrt(dist_k)
    body_normals = igl.per_vertex_normals(body_verts, body_faces)  
    body_normals = body_normals / np.linalg.norm(body_normals, axis=1).reshape(-1, 1)
    offset = dist_init - dist_k
    offset[offset < 0] = 0
    body_verts = body_verts + body_normals * offset.reshape(-1, 1) * mask.reshape(-1, 1)
    
    betas = np.load('./python/mesh/beta_1.npy')
    pose = np.load('./python/mesh/pose.npy')
    rotation_matrix = np.load('./python/mesh/rotations1.npy')[0]
    transl = np.load('./python/mesh/transs1.npy')[0]
    betas = torch.tensor(betas, dtype=torch.float32, device='cuda')
    body_pose = torch.tensor(pose, dtype=torch.float32, device='cuda')
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device='cuda')
    transl = torch.tensor(transl, dtype=torch.float32, device='cuda')
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    body_verts = torch.tensor(body_verts, dtype=torch.float32, device='cuda')
    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
    # body_pose.requires_grad = True
    betas.requires_grad = True
    # quaternion.requires_grad = True
    # transl.requires_grad = True
    optimizer = torch.optim.Adam([betas], lr=1e-3)
    for i in range(1500):
        if i > 800:
            optimizer = torch.optim.Adam([betas], lr=1e-3)
        optimizer.zero_grad()
        out = smpl_model(betas=betas, body_pose=body_pose, return_verts=True)
        body_verts_opt = out['vertices'].squeeze(0)
        body_verts_opt = torch.matmul(body_verts_opt, quaternion_to_rotation_matrix(quaternion).T) + transl
        if i < 10000:
            loss = (torch.norm(body_verts_opt - body_verts, dim=1) * mask).mean() * 1000
        else:
            loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        body_loss = distance_loss(0.003, body_verts_opt, v_init, f)
        print(loss.requires_grad, loss.grad_fn)
        loss += body_loss * 1000
        print(loss.requires_grad, loss.grad_fn)
        # if (i > 100):
        #     loss += body_loss * 30000
        print("body_loss = ", body_loss * 1000)
        print("i = ", i, "loss = ", loss)
        print("torch.norm(body_verts_opt - body_verts_init, dim=1).mean() = ", torch.norm(body_verts_opt - body_verts_init, dim=1).mean() * 1000)
        loss.backward()
        optimizer.step()

    return body_verts_opt.detach().cpu().numpy()

joint_mapper = JointMapper(smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True, use_face_contour=False, openpose_format='coco25'))
smpl_model = smplx.create("/root/cloth-body-reconstruction/smplx/models", model_type="smplx", joint_mapper=joint_mapper)
output = smpl_model(return_verts=True)
beta = output['betas']
pose = output['body_pose']
orient = output['global_orient']
transl = output['transl']

body_mesh = trimesh.load_mesh('./python/mesh/body_mesh1/body_0.obj')
body_verts = body_mesh.vertices
body_faces = body_mesh.faces
goal_mesh = trimesh.load_mesh('./python/mesh/body_mesh1/cloth.obj')
goal_verts = goal_mesh.vertices
faces = goal_mesh.faces    
squares = np.cross(goal_verts[faces][:, 1, :] - goal_verts[faces][:, 0, :], goal_verts[faces][:, 2, :] - goal_verts[faces][:, 0, :])
squares = np.linalg.norm(squares, axis=1) / 2
gt_edges1 = goal_verts[faces][:, 1, :] - goal_verts[faces][:, 0, :]
gt_edges2 = goal_verts[faces][:, 2, :] - goal_verts[faces][:, 0, :]
gt_edges3 = goal_verts[faces][:, 2, :] - goal_verts[faces][:, 1, :]
verts_square = np.zeros(goal_verts.shape[0])
for i in range(faces.shape[0]):
    for j in range(faces.shape[1]):
        verts_square[faces[i][j]] += squares[i] / 3
# read obj mesh
loss_list = []
dist_list = []

# simulate()

iterations = 20
for iter in range(iterations):
    init_mesh = trimesh.load_mesh('./output/cloth_surface0.obj')
    rest_k_mesh = trimesh.load_mesh('./python/mesh/opt_mesh_dress.obj')
    k_mesh = trimesh.load_mesh('./output/cloth_surface100.obj') 
    # save loss

    init_verts = init_mesh.vertices
    rest_k_verts = rest_k_mesh.vertices
    k_verts = k_mesh.vertices
    all_edge_length = np.linalg.norm(k_verts[faces][:, 0, :] - k_verts[faces][:, 1, :], axis=1) + np.linalg.norm(k_verts[faces][:, 1, :] - k_verts[faces][:, 2, :], axis=1) + np.linalg.norm(k_verts[faces][:, 2, :] - k_verts[faces][:, 0, :], axis=1)
    all_edge_length = np.sum(all_edge_length)
    print("all_edge_length = ", all_edge_length)
    dist = torch.norm(torch.tensor(k_verts, dtype=torch.float32, device='cuda') - torch.tensor(init_verts, dtype=torch.float32, device='cuda'), dim=1)
    print("dist = ", torch.sum(dist))
    dist_list.append(torch.sum(dist))
    # body_verts = optimize_body(k_verts, init_verts, faces, body_verts, body_faces, smpl_model, beta, pose, orient, transl)
    v_opt, loss = optimize(rest_k_verts, k_verts, goal_verts, faces, iter)
    # trimesh.Trimesh(vertices=body_verts, faces=body_faces).export('./python/mesh/body_smooth_212.obj')
    loss = loss.cpu().numpy()
    loss = np.array(loss, dtype=np.float64)
    print("loss = ", loss)
    loss_list.append(loss)
    # write obj mesh
    init_center = np.mean(init_verts, axis=0)
    opt_center = np.mean(v_opt, axis=0)
    v_opt += init_center - opt_center
    # print("v_opt - v_k = ", v_opt - k_verts)
    trimesh.Trimesh(vertices=v_opt, faces=faces).export('./python/mesh/opt_mesh_dress.obj')
    # trimesh.Trimesh(vertices=body_verts, faces=body_faces).export('./python/mesh/body_smooth_212.obj')
    simulate()

print(loss_list)
print(dist_list)
losses = np.array(loss_list)
distances = np.array(dist_list)
np.save('./python/mesh/losses.npy', losses)
np.save('./python/mesh/distances.npy', distances)