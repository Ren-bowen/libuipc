import torch

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    rotation_matrix = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w]),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w]),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y])
    ], dim=0)
    return rotation_matrix

def rotation_matrix_to_quaternion(R):
    w = torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4*w)
    y = (R[0, 2] - R[2, 0]) / (4*w)
    z = (R[1, 0] - R[0, 1]) / (4*w)
    return torch.stack([w, x, y, z])