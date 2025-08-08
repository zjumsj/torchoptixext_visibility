import argparse
import numpy as np
import cv2

import torch
import torchoptixext_visibility

from obj_parser import ObjParser

def create_pinhole_camera(pos, right, head, H, W, fovy=60.):
    forward = torch.cross(right, -head)
    basis = torch.stack([
        right / torch.linalg.norm(right),
        -head / torch.linalg.norm(head),
        forward / torch.linalg.norm(forward)
    ],dim=0)

    fov_rad = fovy * np.pi / 180.0
    half_height = np.tan(fov_rad / 2.0)
    aspect_ratio = W / H
    half_width = aspect_ratio * half_height

    jj, ii = torch.meshgrid(
        torch.arange(0, H, 1, device=pos.device, dtype=torch.float32),
        torch.arange(0, W, 1, device=pos.device, dtype=torch.float32),
        indexing = 'ij'
    )
    dir_cam = torch.stack([
        (ii + 0.5) / W * half_width * 2 - half_width,
        (jj + 0.5) / H * half_height * 2 - half_height,
        torch.ones_like(ii)
    ],dim=-1)
    rays_d = dir_cam @ basis
    #rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = torch.ones([H, W, 3], device=pos.device, dtype=torch.float32) * pos

    return rays_o, rays_d

def create_envmap_camera(pos, right, head, H, W):
    backward = torch.cross(right, head)
    basis = torch.stack([
        right / torch.linalg.norm(right),
        head / torch.linalg.norm(head),
        backward / torch.linalg.norm(backward)
    ],dim=0)

    u = torch.arange(0, W, 1, device=pos.device, dtype=torch.float32)
    u = (u + 0.5) / W * (2 * np.pi)
    v = torch.arange(0, H, 1, device=pos.device, dtype=torch.float32)
    v = (v + 0.5) / H * np.pi
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    x = torch.sin(vv) * torch.cos(uu)
    y = torch.cos(vv)
    z = torch.sin(vv) * torch.sin(uu)
    dir_cam = torch.stack([x, y, z], dim=-1)

    rays_d = dir_cam @ basis
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = torch.ones([H, W, 3], device=pos.device, dtype=torch.float32) * pos

    return rays_o, rays_d

def load_mesh(filename, device):
    parser = ObjParser()
    parser.load(filename)
    v = parser.v
    f = parser.f
    v = np.asarray(v,dtype=np.float32)
    f = np.asarray(f,dtype=np.int32)
    v = torch.from_numpy(v).to(device)
    f = torch.from_numpy(f).to(device)
    torchoptixext_visibility.optixDist_load_mesh([v],[f])

def to_image(x):
    non_inf_value = x[torch.logical_not(torch.isinf(x))]
    max_value = non_inf_value.max()
    v = torch.clip(x / max_value, 0., 1.)
    v = torch.round(v * 255.).to(torch.uint8)
    return v.cpu().detach().numpy()

def renderDepthMap(camera_type,H,W,fovy):

    device = "cuda:0"
    torchoptixext_visibility.optixDist_init(device)

    #########################################################
    # Load first scene
    load_mesh("./assets/Town/Town4uO3DKBCRl1iutjJ.obj",device)

    # Render first frame
    theta = -1.1708
    pos = torch.tensor([6., 1.8, 1.2],dtype=torch.float32, device=device)
    right = torch.tensor([np.cos(theta), 0., np.sin(theta)], dtype=torch.float32, device=device)
    head = torch.tensor([0., 1., 0.],dtype=torch.float32, device=device)
    if camera_type == 'pinhole_camera':
        rays_o, rays_d = create_pinhole_camera(
            pos=pos, right=right, head=head,
            H=H, W=W, fovy=fovy
        )
    else:
        rays_o, rays_d = create_envmap_camera(
            pos=pos, right=right, head=head,
            H=H, W=W
        )

    dist = torchoptixext_visibility.optixDist_trace(rays_o, rays_d)
    img = to_image(dist)
    cv2.imwrite('envmap_0.png', img)
    print('Dump envmap_0.png')

    # Render second frame (change view)
    theta = -2.65
    pos = torch.tensor([0., 1.7, 1.],dtype=torch.float32, device=device)
    right = torch.tensor([np.cos(theta), 0., np.sin(theta)], dtype=torch.float32, device=device)
    head = torch.tensor([0., 1., 0.], dtype=torch.float32, device=device)
    if camera_type == 'pinhole_camera':
        rays_o, rays_d = create_pinhole_camera(
            pos=pos, right=right, head=head,
            H=H, W=W, fovy=fovy
        )
    else:
        rays_o, rays_d = create_envmap_camera(
            pos=pos, right=right, head=head,
            H=H, W=W
        )

    dist = torchoptixext_visibility.optixDist_trace(rays_o, rays_d)
    img = to_image(dist)
    cv2.imwrite('envmap_1.png', img)
    print('Dump envmap_1.png')

    #########################################################
    # Change scene
    load_mesh("./assets/MC/scene.obj",device)
    # Render a new frame with a different scene
    theta = -0.6283
    pos = torch.tensor([-3., 19.7, 3.],dtype=torch.float32, device=device)
    right = torch.tensor([np.cos(theta), 0., np.sin(theta)], dtype=torch.float32, device=device)
    head = torch.tensor([0., 1., 0.], dtype=torch.float32, device=device)
    if camera_type == 'pinhole_camera':
        rays_o, rays_d = create_pinhole_camera(
            pos=pos, right=right, head=head,
            H=H, W=W, fovy=fovy
        )
    else:
        rays_o, rays_d = create_envmap_camera(
            pos=pos, right=right, head=head,
            H=H, W=W
        )

    dist = torchoptixext_visibility.optixDist_trace(rays_o, rays_d)
    img = to_image(dist)
    cv2.imwrite('envmap_2.png', img)
    print('Dump envmap_2.png')

    #### clean
    torchoptixext_visibility.optixDist_destroy_mesh(device)
    torchoptixext_visibility.optixDist_destroy(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render depth map.')
    parser.add_argument('-H','--height',type=int, default=720, help='camera width')
    parser.add_argument('-W','--width',type=int, default=1280, help='camera height')
    parser.add_argument('--camera_type', type=str, default='pinhole_camera', choices=['envmap_camera','pinhole_camera'], help='camera type')
    parser.add_argument('--fovy', type=float, default=60, help='camera fovy (only used in pinhole camera)')
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print(unknown)
        exit(-1)
    renderDepthMap(
        args.camera_type, args.height, args.width, args.fovy
    )