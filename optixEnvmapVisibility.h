#pragma once

namespace optixEnvmapVisibility{

    struct LaunchParams {
        unsigned int width;
        unsigned int height;
        unsigned int envmap_width;
        unsigned int envmap_height;
        unsigned int * frame_buffer;
        float3 * rays_o;
        float3 * rays_d;
        float rotray[9]; // row major
        OptixTraversableHandle handle;
    };

    struct RayGenData
	{
		// No data needed
	};

    struct MissData
	{
		// No data needed
	};

    struct HitGroupData
	{
		// No data needed
	};


}