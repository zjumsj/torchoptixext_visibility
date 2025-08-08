/*
	source list

	binding1.cpp
	PyOptixTriangle.cpp
	PyOptixTrianglecu.cu

	CUDAOutputBuffer.h
	Exception.h
	myTools.cpp
	myTools.h
	OptixUtil.cpp
	OptixUtil.h

*/
#include <string>
#include <torch/extension.h>

std::string gb_path;

torch::Tensor run_sample_code();
void set_path(const std::string s) {
	gb_path = s;
}

//-----------------------------------
void optixDist_init();
void optixDist_load_mesh(
	std::vector<torch::Tensor> vertex_group,
	std::vector<at::optional<at::Tensor>> index_group
);
torch::Tensor optixDist_trace(
	torch::Tensor rays_o,
	torch::Tensor rays_d
);
void optixDist_destroy_mesh();
void optixDist_destroy();
//-----------------------------------
void optixEnvmapVisibility_init(int shader, int fullupdate_step);
void optixEnvmapVisibility_load_mesh(
	std::vector<torch::Tensor> vertex_group,
	std::vector<at::optional<at::Tensor>> index_group
);
torch::Tensor optixEnvmapVisibility_trace(
	torch::Tensor rays_o, at::optional<at::Tensor> rays_d,
	torch::Tensor rot_ray,
	unsigned int envmap_width, unsigned int envmap_height
);
void optixEnvmapVisibility_destroy_mesh();
void optixEnvmapVisibility_destroy();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("set_path", &set_path, "set package path");
	m.def("run_sample_code", &run_sample_code, "run OptixTriangle sample code.");

	m.def("optixDist_init", &optixDist_init, "optixDist init context");
	m.def("optixDist_destroy", &optixDist_destroy, "optixDist destroy context");
	m.def("optixDist_load_mesh", &optixDist_load_mesh, "optixDist load mesh");
	m.def("optixDist_destroy_mesh", &optixDist_destroy_mesh, "optixDist destroy mesh");
	m.def("optixDist_trace", &optixDist_trace, "optixDist run trace after loading mesh");

	m.def("optixEnvmapVisibility_init", &optixEnvmapVisibility_init, "optixEnvmapVisibility init context");
	m.def("optixEnvmapVisibility_destroy", &optixEnvmapVisibility_destroy, "optixEnvmapVisibility destroy context");
	m.def("optixEnvmapVisibility_load_mesh", &optixEnvmapVisibility_load_mesh, "optixEnvmapVisibility load mesh");
	m.def("optixEnvmapVisibility_destroy_mesh", &optixEnvmapVisibility_destroy_mesh, "optixEnvmapVisibility destroy mesh");
	m.def("optixEnvmapVisibility_trace", &optixEnvmapVisibility_trace, "optixEnvmapVisibility run trace after loading mesh");

}
