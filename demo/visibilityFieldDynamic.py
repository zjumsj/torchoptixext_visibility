'''
    Demo: Create Visibility Field for a 2D light probe grid in a dynamic scene
'''
import os
import argparse
import numpy as np
import cv2

import torch
import torchoptixext_visibility

# Creating a squirming donut
def generate_torus(R, r, N, M, t=0.):
    vertices = np.zeros((N * M, 3))
    # Generate vertices
    for i in range(N):  # Loop through major circle segments
        # Angle around the major circle
        theta = 2 * np.pi * i / N
        for j in range(M):  # Loop through minor circle segments
            # Angle around the minor circle
            phi = 2 * np.pi * j / M
            # Calculate vertex position
            x = (R + r * np.cos(phi)) * np.cos(theta)
            z = (R + r * np.cos(phi)) * np.sin(theta)
            #y = r * np.sin(phi)
            y = r * np.sin(phi) + 0.3 * r * np.sin(5 * theta + 3 * t)
            # Store vertex
            vertices[i * M + j] = [x, y, z]
    # Initialize face list (2 triangles per quad)
    faces = np.zeros((2 * N * M, 3), dtype=int)
    face_index = 0
    # Generate triangular faces
    for i in range(N):  # Loop through major circle segments
        for j in range(M):  # Loop through minor circle segments
            # Calculate next indices with wrapping
            i_next = (i + 1) % N
            j_next = (j + 1) % M
            # Current quad vertex indices
            v00 = i * M + j  # Current vertex
            v01 = i * M + j_next  # Next minor segment
            v10 = i_next * M + j  # Next major segment
            v11 = i_next * M + j_next  # Next major and minor segment
            # Create two triangles per quad
            # Triangle 1
            faces[face_index] = [v00, v01, v11]
            face_index += 1
            # Triangle 2
            faces[face_index] = [v00, v11, v10]
            face_index += 1

    if True: # slightly rotate it.
        rot_mat = np.array([
            [1.,0., 0.],
            [0.,0.86603, 0.5],
            [0.,-0.5, 0.86603]
        ])
        vertices = vertices @ rot_mat

    return vertices, faces

def to_image(arr, envmap_H, envmap_W, row, col):
    # arr N x P
    P = arr.shape[-1]
    arr_uint32 = arr.view(np.uint32)
    bit_mask = (1 << np.arange(32, dtype=np.uint32)).reshape(-1,1)
    bits = ((arr_uint32[:,np.newaxis,:] & bit_mask) != 0).astype(np.uint8) # Nx32xP
    bits = np.reshape(bits, [-1, P])[:envmap_H * envmap_W] # (tHxtW)xP
    bits = np.reshape(bits, [envmap_H, envmap_W, row, col]) # tHxtWxHxW
    bits = np.transpose(bits, [2,0,3,1]) # HxtHxWxtW
    bits = np.reshape(bits, [row * envmap_H, col * envmap_W])
    return bits * 255

def renderVisibilityBitfieldDynamic(
    envmap_H, envmap_W, xmin, xmax, zmin, zmax, y, row, col, testFPS
):
    device = "cuda:0"
    # fullupdate_step > 0: hybrid rebuilding - rebuild from scratch every 10 steps, otherwise perform fast dynamic updates
    # Prerequisites: The updated mesh must maintain identical topology with moderate vertex changes.
    torchoptixext_visibility.optixEnvmapVisibility_init(device, shader=1, fullupdate_step=10)

    if not testFPS:
        os.makedirs("visibilityFieldDynamic", exist_ok=True)
        print('Dump in folder visibilityFieldDynamic')

    z, x = torch.meshgrid(
        torch.linspace(xmin, xmax, col, device=device, dtype=torch.float32),
        torch.linspace(zmin, zmax, row, device=device, dtype=torch.float32),
        indexing = 'ij'
    )
    probe_center = torch.stack([
        x,
        torch.ones_like(x) * y,
        z
    ],dim=-1)
    rot_ray_transform = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],dtype=torch.float32,device=device)

    N = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    mesh_list = [generate_torus(0.3, 0.08, 40, 20, t=j * (1./24.)) for j in range(N)]
    v_list = [torch.from_numpy(o[0]).float().to(device) for o in mesh_list]
    f_list = [torch.from_numpy(o[1]).int().to(device) for o in mesh_list]

    for i_iter in range(N):

        if testFPS and i_iter == 10: # Warmup
            start_event.record()

        ## Creating mesh on CPU is slow and affects measurement accuracy. Here we pre-create the mesh and store it on GPU.

        #v, f = generate_torus(0.3, 0.08, 40, 20, t=i_iter * (1./24.))
        #v = torch.from_numpy(v).float().to(device)
        #f = torch.from_numpy(f).int().to(device)

        v = v_list[i_iter]
        f = f_list[i_iter]
        torchoptixext_visibility.optixEnvmapVisibility_load_mesh([v],[f])
        bitfields = torchoptixext_visibility.optixEnvmapVisibility_trace(
            rays_o=probe_center.view(-1, 3),
            rays_n=None,
            rot_ray=rot_ray_transform,
            envmap_width=envmap_W,
            envmap_height=envmap_H
        )

        if not testFPS:
            bitfields = bitfields.cpu().detach().numpy()
            img = to_image(bitfields, envmap_H, envmap_W, row, col)
            cv2.imwrite("visibilityFieldDynamic/%05d.png" % i_iter, img)

    if testFPS:
        end_event.record()
        torch.cuda.synchronize()
        cuda_time = start_event.elapsed_time(end_event)
        ms_per_frame = cuda_time / (N-10)
        fps = 1000 / ms_per_frame
        print('%.4f ms per frame, fps %.2f' % (ms_per_frame, fps))

    # clean
    torchoptixext_visibility.optixEnvmapVisibility_destroy_mesh(device)
    torchoptixext_visibility.optixEnvmapVisibility_destroy(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute visibility in dynamic scene.')

    parser.add_argument('--envmap_H',type=int, default=128, help='envmap height')
    parser.add_argument('--envmap_W',type=int, default=256, help='envmap width')

    parser.add_argument('--xmin', type=float, default=-0.4, help='2D gird xmin')
    parser.add_argument('--xmax', type=float, default=0.4, help='2D grid xmax')
    parser.add_argument('--zmin', type=float, default=-0.4, help='2D grid zmin')
    parser.add_argument('--zmax', type=float, default=0.4, help='2D grid zmax')
    parser.add_argument('--y', type=float, default=0., help='2D grid y')

    parser.add_argument('--row', type=int, default=3, help='row number for light probe 2D grid')
    parser.add_argument('--col', type=int, default=3, help='col number for light probe 2D grid')

    parser.add_argument('--testFPS', action='store_true', default=False, help='enable FPS testing')

    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(unknown)
        exit(-1)
    renderVisibilityBitfieldDynamic(
        args.envmap_H, args.envmap_W,
        args.xmin, args.xmax, args.zmin, args.zmax, args.y,
        args.row, args.col, args.testFPS
    )

