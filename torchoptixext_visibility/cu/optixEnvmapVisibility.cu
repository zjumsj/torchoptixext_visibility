
#include <optix.h>

#include "optixEnvmapVisibility.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

#include "config.h"

extern "C" {
	__constant__ optixEnvmapVisibility::LaunchParams params;
}

static __forceinline__ __device__ float3 getLightDir(int iy, int ix, int W, int H)
{
    float v = (float(iy) + 0.5f) / H;
    float u = (float(ix) + 0.5f) / W;
    float phi = u * M_PIf * 2;
    float theta = (v - 0.5f) * M_PIf;
    float sin_theta, cos_theta;
    float sin_phi, cos_phi;
    sincosf(theta, &sin_theta, &cos_theta);
    sincosf(phi, &sin_phi, &cos_phi);
    float3 c;
    c.y = sin_theta;
    c.x = cos_theta * cos_phi;
    c.z = cos_theta * sin_phi;
    return c;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.rays_o[idx.x];

    unsigned int p0;
    unsigned int envmap_width = params.envmap_width;
    unsigned int envmap_height = params.envmap_height;
    unsigned int envmap_elem = envmap_width * envmap_height;
    const float * l_mat = params.rotray;
    
    unsigned int buffer = 0;
    unsigned int slot = 0;
    for(unsigned int i_texel = 0; i_texel < envmap_elem; i_texel++){
        unsigned int ix = i_texel % envmap_width;
        unsigned int iy = i_texel / envmap_width;
        unsigned int offset = i_texel % 32;
#ifdef LEFT_TOP_AS_ORIGIN
        iy = envmap_height - 1 - iy;
#endif
        float3 l_dir = getLightDir(iy, ix, envmap_width, envmap_height);
        float3 rot_l_dir;
        rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
        rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
        rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
        unsigned int p0;
        optixTrace(
            params.handle,
            ray_origin,
            rot_l_dir,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0);
        buffer = (buffer | (p0 << offset));

        if((i_texel % 32) == 31){
            params.frame_buffer[slot * params.width + idx.x] = buffer;
            ++slot;
            buffer = 0;
        }
    }
    if((envmap_elem % 32) != 0){
        // write another time
        params.frame_buffer[slot * params.width + idx.x] = buffer;
    }
}

extern "C" __global__ void __raygen__rg1()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.rays_o[idx.x];

    unsigned int p0;
    unsigned int envmap_width = params.envmap_width;
    unsigned int envmap_height = params.envmap_height;
    unsigned int envmap_elem = envmap_width * envmap_height;
    const float * l_mat = params.rotray;

    unsigned int buffer = 0;
    for(unsigned int i = 0; i < 32; i++ ){
        unsigned int i_texel = idx.y * 32 + i;
        if(i_texel < envmap_elem){
            unsigned int ix = i_texel % envmap_width;
            unsigned int iy = i_texel / envmap_width;
            //unsigned int offset = i_texel % 32;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = envmap_height - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, envmap_width, envmap_height);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            unsigned int p0;
            optixTrace(
                params.handle,
                ray_origin,
                rot_l_dir,
                0.0f,                // Min intersection distance
                1e16f,               // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask(255), // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                p0);
            buffer = (buffer | (p0 << i));
        }
    }
    params.frame_buffer[idx.y * params.width + idx.x] = buffer;
}

extern "C" __global__ void __raygen__rg2()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin = params.rays_o[idx.x];

    unsigned int p0;
    unsigned int envmap_width = params.envmap_width;
    unsigned int envmap_height = params.envmap_height;
    unsigned int envmap_elem = envmap_width * envmap_height;
    const float * l_mat = params.rotray;

    unsigned int buffer = 0;
    for(unsigned int i = 0; i < 8; i++){
        unsigned int i_texel = idx.y * 8 + i;
        if(i_texel < envmap_elem){
            unsigned int ix = i_texel % envmap_width;
            unsigned int iy = i_texel / envmap_width;
            unsigned int offset = i_texel % 32;
    #ifdef LEFT_TOP_AS_ORIGIN
                iy = envmap_height - 1 - iy;
    #endif
            float3 l_dir = getLightDir(iy, ix, envmap_width, envmap_height);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            unsigned int p0;
            optixTrace(
                params.handle,
                ray_origin,
                rot_l_dir,
                0.0f,                // Min intersection distance
                1e16f,               // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask(255), // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                p0);
            buffer = (buffer | (p0 << offset));
        }
    }
    atomicOr(&params.frame_buffer[(idx.y / 4) * params.width + idx.x], buffer);
}

extern "C" __global__ void __raygen__rg3()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 rays_o = params.rays_o[idx.x];
    float3 rays_n = params.rays_d[idx.x];
    float3 ray_origin = rays_o + rays_n * 1e-5f;

    unsigned int p0;
    unsigned int envmap_width = params.envmap_width;
    unsigned int envmap_height = params.envmap_height;
    unsigned int envmap_elem = envmap_width * envmap_height;
    const float* l_mat = params.rotray;
    const float dot_thres = 0.f;
    //const float dot_thres = -0.1f;

    unsigned int buffer = 0;
    for (unsigned int i = 0; i < 32; i++) {
        unsigned int i_texel = idx.y * 32 + i;
        if (i_texel < envmap_elem) {
            unsigned int ix = i_texel % envmap_width;
            unsigned int iy = i_texel / envmap_width;
            //unsigned int offset = i_texel % 32;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = envmap_height - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, envmap_width, envmap_height);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            unsigned int p0 = 0;
            if (dot(rot_l_dir, rays_n) >= dot_thres) {
                optixTrace(
                    params.handle,
                    ray_origin,
                    rot_l_dir,
                    0.0f,                // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(255), // Specify always visible
                    //OPTIX_RAY_FLAG_NONE,
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0,                   // SBT offset   -- See SBT discussion
                    1,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    p0);
            }
            buffer = (buffer | (p0 << i));
        }
    }
    params.frame_buffer[idx.y * params.width + idx.x] = buffer; 
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(0);
}
