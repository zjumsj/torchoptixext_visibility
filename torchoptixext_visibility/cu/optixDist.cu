#include <optix.h>

#include "optixDist.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" {
	__constant__ optixDist::LaunchParams params;
}

static __forceinline__ __device__ void setPayload(float p) {
	optixSetPayload_0(float_as_int(p));
}

extern "C" __global__ void __raygen__rg()
{
	// Lookup our location within the launch grid
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	// Map our launch idx to a screen location and create a ray from the camera
	// location through the screen

	//float3 ray_origin, ray_direction;
	//computeRay(idx, dim, ray_origin, ray_direction);
	float3 ray_origin = params.rays_o[idx.y * params.width + idx.x];
	float3 ray_direction = params.rays_d[idx.y * params.width + idx.x];
	//float3 ray_origin = { 0.f,0.f,-5.f };
	//float3 ray_direction = { 0.f,0.f,1.f };
	
	//unsigned int p0, p1, p2;
	unsigned int p0;
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		0.0f,                // Min intersection distance
		1e16f,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		0,                   // SBT offset   -- See SBT discussion
		1,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		p0);

	float result;
	result = int_as_float(p0);

	params.frame_buffer[idx.y * params.width + idx.x] = result;
}

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

extern "C" __global__ void __miss__ms()
{
	setPayload(CUDART_INF_F);
}

extern "C" __global__ void __closesthit__ch()
{
	float t = optixGetRayTmax();
	setPayload(t);
}
