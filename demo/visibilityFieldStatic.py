'''
    Demo: Create Visibility Field for a 2D light probe grid in a static scene
'''
import argparse
import numpy as np
import cv2

import torch
import torchoptixext_visibility

from obj_parser import ObjParser

def load_mesh(filename, device):
    parser = ObjParser()
    parser.load(filename)
    v = parser.v
    f = parser.f
    v = np.asarray(v,dtype=np.float32)
    f = np.asarray(f,dtype=np.int32)
    v = torch.from_numpy(v).to(device)
    f = torch.from_numpy(f).to(device)
    torchoptixext_visibility.optixEnvmapVisibility_load_mesh([v],[f])

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

def renderVisibilityBitfield(
    envmap_H, envmap_W, xmin, xmax, zmin, zmax, y, row, col
):
    device = "cuda:0"
    torchoptixext_visibility.optixEnvmapVisibility_init(device,shader=1)

    # load scene
    load_mesh("./assets/Town/Town4uO3DKBCRl1iutjJ.obj", device)
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
    bitfields = torchoptixext_visibility.optixEnvmapVisibility_trace(
        rays_o = probe_center.view(-1,3),
        rays_n = None,
        rot_ray = rot_ray_transform,
        envmap_width = envmap_W,
        envmap_height = envmap_H
    )
    bitfields = bitfields.cpu().detach().numpy()
    img = to_image(bitfields, envmap_H, envmap_W, row, col)
    cv2.imwrite('visibilityFieldStatic.png', img)
    print('Dump visibilityFieldStatic.png')

    # clean
    torchoptixext_visibility.optixEnvmapVisibility_destroy_mesh(device)
    torchoptixext_visibility.optixEnvmapVisibility_destroy(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute visibility in static scene.')

    parser.add_argument('--envmap_H',type=int, default=64, help='envmap height')
    parser.add_argument('--envmap_W',type=int, default=128, help='envmap width')

    parser.add_argument('--xmin', type=float, default=-4., help='2D grid xmin')
    parser.add_argument('--xmax', type=float, default=4., help='2D grid xmax')
    parser.add_argument('--zmin', type=float, default=-4, help='2D grid zmin')
    parser.add_argument('--zmax', type=float, default=4, help='2D grid zmax')
    parser.add_argument('--y', type=float, default=1.8, help='2D grid y')

    parser.add_argument('--row', type=int, default=10, help='row number for light probe 2D grid')
    parser.add_argument('--col', type=int, default=10, help='col number for light probe 2D grid')

    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(unknown)
        exit(-1)
    renderVisibilityBitfield(
        args.envmap_H, args.envmap_W,
        args.xmin, args.xmax, args.zmin, args.zmax, args.y,
        args.row, args.col
    )