import os
import torch
import numpy as np
import trimesh
from clothed_human import simulate
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    cot = torch.zeros_like(f)
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
        T_ig = torch.matmul(torch.inverse(A), T_goal)
        # print("torch.det(A) = ", torch.det(A))
        # print("A = ", A)
        # since T_init and T are both upper triangular matrix, A and T_ig are also upper triangular matrix

        # transfer back to 3d coordinate
        u_ig = torch.tensor([T_ig[0][0], 0, 0], device='cuda')
        v_ig = torch.tensor([T_ig[0][1], T_ig[1][1], 0], device='cuda')
        cot[i][2] = v_ig[0] / v_ig[1]
        cot[i][1] = (u_ig[0] - v_ig[0]) / v_ig[1]
        cos_i_0 = (u_ig * u_ig + v_ig * v_ig - (u_ig - v_ig) * (u_ig - v_ig)) / (2 * torch.norm(u_ig) * torch.norm(v_ig))
        cot[i][0] = cos_i_0 / torch.sqrt(1 - cos_i_0 ** 2)
        
        
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
        # local optimization
        # optimize the rotation matrix L by minimizing ||u - L @ u_ig||^2 + ||v - L @ v_ig||^2 + ||(u - v) - (L @ u_ig - L @ v_ig)||^2
        # since L is a rotation matrix, we have L @ L^T = I, then the problem is equal to that:
        # minimize -2 * (u_ig.T @ L @ u + v_ig.T @ L @ v + (u_ig - v_ig).T @ L @ (u - v)) <=>
        # minimize -L :: X, where X = u @ u_ig.T + v @ v_ig.T + (u - v) @ (u_ig - v_ig).T <=>
        # minimize ||X - L||^2, where ||.|| is the Frobenius norm.
        # This can be solved by Procrustes analysis.
        # Do SVD on X = U @ S @ V.T, then L = U @ V.T
        X = torch.outer(u_init, u_ig) + torch.outer(v_init, v_ig) + torch.outer(u_init - v_init, u_ig - v_ig)
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

    edges_init = torch.norm(X_init[:, 0] - X_init[:, 1], dim=1) + torch.norm(X_init[:, 1] - X_init[:, 2], dim=1) + torch.norm(X_init[:, 2] - X_init[:, 0], dim=1)
    edges_k = torch.norm(X_k[:, 0] - X_k[:, 1], dim=1) + torch.norm(X_k[:, 1] - X_k[:, 2], dim=1) + torch.norm(X_k[:, 2] - X_k[:, 0], dim=1)
    edges_goal = torch.norm(X_goal[:, 0] - X_goal[:, 1], dim=1) + torch.norm(X_goal[:, 1] - X_goal[:, 2], dim=1) + torch.norm(X_goal[:, 2] - X_goal[:, 0], dim=1)
    edges_ig = torch.norm(X_ig[:, 0] - X_ig[:, 1], dim=1) + torch.norm(X_ig[:, 1] - X_ig[:, 2], dim=1) + torch.norm(X_ig[:, 2] - X_ig[:, 0], dim=1)
    length = torch.tensor([torch.sum(edges_init), torch.sum(edges_k), torch.sum(edges_goal), torch.sum(edges_ig)], device='cuda')
    length = length.cpu().numpy()
    # save the edges
    edges_rate_k_goal = edges_k / edges_goal
    edges_rate_ig_goal = edges_ig / edges_goal 
    edges_i = edges_goal * edges_goal / edges_k
    rate_i = edges_goal / edges_k
    os.makedirs('/root/libuipc/python/output/egde_length', exist_ok=True)
    np.savetxt('/root/libuipc/python/output/egde_length/edges_rate_k_goal' + str(iter) + '.txt', edges_rate_k_goal.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/edges_rate_ig_goal' + str(iter) + '.txt', edges_rate_ig_goal.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/edges_i' + str(iter) + '.txt', edges_i.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/rate_i' + str(iter) + '.txt', rate_i.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/edges_init' + str(iter) + '.txt', edges_init.cpu().numpy()) 
    np.savetxt('/root/libuipc/python/output/egde_length/edges_k' + str(iter) + '.txt', edges_k.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/edges_goal' + str(iter) + '.txt', edges_goal.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/edges_ig' + str(iter) + '.txt', edges_ig.cpu().numpy())
    np.savetxt('/root/libuipc/python/output/egde_length/length' + str(iter) + '.txt', length)
    # save the edge length sum


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
    X_ig = X_ig.cpu().numpy()

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(3):
                Q_sparse[3 * f[i][j] + k, 3 * f[i][j] + k] += 2
                Q_sparse[3 * f[i][j] + k, 3 * f[i][(j + 1) % 3] + k] += -1
                Q_sparse[3 * f[i][j] + k, 3 * f[i][(j + 2) % 3] + k] += -1
                p[3 * f[i][j] + k] += 2 * X_ig[i][j][k] - 1 * X_ig[i][(j + 1) % 3][k] - 1 * X_ig[i][(j + 2) % 3][k]

    Q_sparse = Q_sparse.tocsr()
    v_opt = spla.spsolve(Q_sparse, p).reshape(-1, 3)
    return v_opt


goal_mesh = trimesh.load_mesh('/root/libuipc/python/mesh/init_mesh.obj')
goal_verts = goal_mesh.vertices
faces = goal_mesh.faces      
# read obj mesh
loss_list = []
for iter in range(8):
    simulate()
    init_mesh = trimesh.load_mesh('/root/libuipc/output/cloth_surface0.obj')
    rest_k_mesh = trimesh.load_mesh('/root/libuipc/python/mesh/opt_mesh.obj')
    k_mesh = trimesh.load_mesh('/root/libuipc/output/cloth_surface100.obj')
    loss = np.linalg.norm(init_mesh.vertices - k_mesh.vertices)
    # save loss
    np.savetxt('/root/libuipc/python/mesh/loss' + str(iter) + '.txt', np.array([loss]))
    print("loss = ", loss)
    loss_list.append(loss)
    init_verts = init_mesh.vertices
    rest_k_verts = rest_k_mesh.vertices
    k_verts = k_mesh.vertices
    all_edge_length = np.linalg.norm(k_verts[faces][:, 0, :] - k_verts[faces][:, 1, :], axis=1) + np.linalg.norm(k_verts[faces][:, 1, :] - k_verts[faces][:, 2, :], axis=1) + np.linalg.norm(k_verts[faces][:, 2, :] - k_verts[faces][:, 0, :], axis=1)
    all_edge_length = np.sum(all_edge_length)
    print("all_edge_length = ", all_edge_length)
    v_opt = optimize(rest_k_verts, k_verts, goal_verts, faces, iter)
    # write obj mesh
    init_center = np.mean(init_verts, axis=0)
    opt_center = np.mean(v_opt, axis=0)
    v_opt += init_center - opt_center
    # print("v_opt - v_k = ", v_opt - k_verts)
    trimesh.Trimesh(vertices=v_opt, faces=faces).export('/root/libuipc/python/mesh/opt_mesh.obj')
print(loss_list)

