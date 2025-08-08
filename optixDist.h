#pragma once

namespace optixDist {


	struct LaunchParams {
		unsigned int width;
		unsigned int height;
		float * frame_buffer;
		float3 * rays_o;
		float3 * rays_d;
		OptixTraversableHandle handle;

	};

	struct RayGenData
	{
		// No data needed
	};


	//struct MissData
	//{
	//	float3 bg_color;
	//};

	struct MissData
	{
		// No data needed
	};


	struct HitGroupData
	{
		// No data needed
	};


}
