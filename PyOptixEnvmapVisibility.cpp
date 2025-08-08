#include "optix.h"
#include "optix_stubs.h"
#if OPTIX_VERSION < 70300
// <= 7.2.x
#include "OptixUtil.h"
#else
#include "optix_stack_size.h"
#endif

#include <cuda_runtime.h>
#include "optixEnvmapVisibility.h"
#include "ExceptionLocal.h"

#include "myTools.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <array>

//#include "optix_function_table_definition.h"

#include <torch/torch.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

extern std::string gb_path;

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<optixEnvmapVisibility::RayGenData>     RayGenSbtRecord;
typedef SbtRecord<optixEnvmapVisibility::MissData>       MissSbtRecord;
typedef SbtRecord<optixEnvmapVisibility::HitGroupData>   HitGroupSbtRecord;

struct TmpBuffer {
	CUdeviceptr cu_data = 0;
	size_t buffersize_in_byte = 0;
	
	void Destroy(){
		if (cu_data) { cudaFree(reinterpret_cast<void*>(cu_data)); cu_data = 0; buffersize_in_byte = 0; }
	}
	~TmpBuffer() { Destroy(); }
	
	void allocate(size_t n) {
		if (buffersize_in_byte < n) {
			printf("reallocate %zu -> %zu\n", buffersize_in_byte, n);
			Destroy();
			cudaMalloc(reinterpret_cast<void**>(&cu_data), n);
			buffersize_in_byte = n;
		}
	}
};

TmpBuffer buffer_tmp_gas, buffer_gas;

struct MeshInfo {
	unsigned int n_vertex = 0;
	unsigned int n_face = 0;
	std::vector<float> vertex;
	std::vector<int> index;
	unsigned int flatten = 0;

	CUdeviceptr cu_vertex = 0;
	CUdeviceptr cu_index = 0;
	OptixTraversableHandle  gas_handle = 0;
	CUdeviceptr   d_gas_output = 0; // should be kept ?
};

struct BuildingContainer {

	int num_subMeshes;
	std::vector<OptixBuildInput> buildInputs;
	std::vector<CUdeviceptr> pointers;
	unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

	void Update(
		std::vector<torch::Tensor> & vertex_group,
		std::vector<at::optional<at::Tensor>> & index_group
	){
		for (size_t i_subindex = 0; i_subindex < num_subMeshes; i_subindex++) {
			OptixBuildInput & triangle_input = buildInputs[i_subindex];
			
			torch::Tensor & loc_vertex = vertex_group[i_subindex];
			at::optional<at::Tensor> & loc_index = index_group[i_subindex];
			
			pointers[i_subindex] = reinterpret_cast<CUdeviceptr>(loc_vertex.data_ptr());
			triangle_input.triangleArray.vertexBuffers = &pointers[i_subindex];

			if(loc_index.has_value()){
				triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(loc_index.value().data_ptr());
			}
			else {
				triangle_input.triangleArray.indexBuffer = 0;
			}
		}
	}

	void Init(
		std::vector<torch::Tensor> & vertex_group,
		std::vector<at::optional<at::Tensor>> & index_group
	){
		num_subMeshes = vertex_group.size();
		buildInputs.resize(num_subMeshes);
		pointers.resize(num_subMeshes);

		for (size_t i_subindex = 0; i_subindex < num_subMeshes; i_subindex++) {
			OptixBuildInput & triangle_input = buildInputs[i_subindex];
			memset(&triangle_input, 0, sizeof(OptixBuildInput));

			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.vertexStrideInBytes = 0;

			torch::Tensor & loc_vertex = vertex_group[i_subindex];
			at::optional<at::Tensor> & loc_index = index_group[i_subindex];
			
			pointers[i_subindex] = reinterpret_cast<CUdeviceptr>(loc_vertex.data_ptr());
			triangle_input.triangleArray.vertexBuffers = &pointers[i_subindex];
			triangle_input.triangleArray.numVertices = loc_vertex.size(0);

			if(loc_index.has_value()){
				triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				triangle_input.triangleArray.numIndexTriplets = loc_index.value().size(0);
				triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(loc_index.value().data_ptr());
			}
			else {
#if OPTIX_VERSION >= 70100
				triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
#endif
				triangle_input.triangleArray.numIndexTriplets = 0;
				triangle_input.triangleArray.indexBuffer = 0;
			}
			triangle_input.triangleArray.flags = &triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;
		}
	}

};

int optix_envmap_visibility_shader;
int optix_envmap_visibility_fullupdate_step;

BuildingContainer building_container;
OptixAccelBufferSizes bak_buffer_size;


struct optixEnvmapVisibilityState {

    OptixDeviceContext context = 0;
    MeshInfo *         mesh = 0;

	CUstream            stream = 0;

	OptixProgramGroup   hitgroup_prog_group = 0;
	OptixProgramGroup   miss_prog_group = 0;
	OptixProgramGroup   raygen_prog_group = 0;
	OptixModule         module = 0;
	OptixPipeline       pipeline = 0;

	OptixShaderBindingTable  sbt = {};

	optixEnvmapVisibility::LaunchParams  params;
	optixEnvmapVisibility::LaunchParams*  d_params;
};

// Avoid namespace collision
namespace optixEnvmapVisibility{

    optixEnvmapVisibilityState m_optixEnvmapVisibilityState;
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

namespace optixEnvmapVisibility {

    OptixDeviceContext OptixObjCreateContext()
	{
		// Initialize CUDA
		CUDA_CHECK(cudaFree(nullptr));

		OPTIX_CHECK(optixInit());

        // Specify context options
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;

        // Associate a CUDA context (and therefore a specific GPU) with this
		// device context
		CUcontext cuCtx = 0;  // zero means take the current context
		OptixDeviceContext context = 0;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
		return context;
    }

	template <typename IntegerType>
	IntegerType roundUp(IntegerType x, IntegerType y)
	{
		return ((x + y - 1) / y) * y;
	}

    void createProgramGroupsAndPipeline( optixEnvmapVisibilityState & state, int shader)
	{
		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixPipelineCompileOptions pipeline_compile_options = {};
		OptixModuleCompileOptions module_compile_options = {};
        // Set compile options
        {
			module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#if OPTIX_VERSION < 70400
			module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // <= Optix 7.3.x
#else
			module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // Optix 7.4.0
#endif

			pipeline_compile_options.usesMotionBlur = false;
			// NO NEED TO BUILD IAS in this proj ?
			pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
			//pipeline_compile_options.numPayloadValues = 3; //
			//pipeline_compile_options.numAttributeValues = 3; //

			pipeline_compile_options.numPayloadValues = 1;
			pipeline_compile_options.numAttributeValues = 2;

            // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
			//pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

			pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
#if OPTIX_VERSION >= 70100
			pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#endif
		}
		// Create Module
		OptixModule & module = state.module;
		{
			std::string ptx;
			const std::string module_name = myTools::cat_path(gb_path.c_str(), "cu/optixEnvmapVisibility.cu");
			ptx = myTools::getPtxString(module_name.c_str(), nullptr, &gb_path);

#if OPTIX_VERSION < 70700
			OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
				state.context,
				&module_compile_options,
				&pipeline_compile_options,
				ptx.c_str(),
				ptx.size(),
				log,
				&sizeof_log,
				&module
			));
#else
            OPTIX_CHECK_LOG(optixModuleCreate(
				state.context,
				&module_compile_options,
				&pipeline_compile_options,
				ptx.c_str(),
				ptx.size(),
				log,
				&sizeof_log,
				&module
			));
#endif
		}
        // Create Program groups
		OptixProgramGroup & raygen_prog_group = state.raygen_prog_group;
		OptixProgramGroup & miss_prog_group = state.miss_prog_group;
		OptixProgramGroup & hitgroup_prog_group = state.hitgroup_prog_group;
		{
			OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
			OptixProgramGroupDesc raygen_prog_group_desc = {};
			raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygen_prog_group_desc.raygen.module = module;
			if (shader == 0)
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
			else if (shader == 1)
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg1";
			else if (shader == 2)
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg2";			
			else
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg3";

			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&raygen_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&raygen_prog_group
			));

			OptixProgramGroupDesc miss_prog_group_desc = {};
			miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			miss_prog_group_desc.miss.module = module;
			miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
			//sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&miss_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&miss_prog_group
			));

			OptixProgramGroupDesc hitgroup_prog_group_desc = {};
			hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroup_prog_group_desc.hitgroup.moduleCH = module;
			hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
			//sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				state.context,
				&hitgroup_prog_group_desc,
				1,   // num program groups
				&program_group_options,
				log,
				&sizeof_log,
				&hitgroup_prog_group
			));
		}
		// Link pipeline
		OptixPipeline & pipeline = state.pipeline;
		{
			const uint32_t    max_trace_depth = 1;
			const uint32_t    max_traversal_depth = 1;
			OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

			OptixPipelineLinkOptions pipeline_link_options = {};
			pipeline_link_options.maxTraceDepth = max_trace_depth;
#if OPTIX_VERSION < 70700
			pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
			//size_t sizeof_log = sizeof(log);
			OPTIX_CHECK_LOG(optixPipelineCreate(
				state.context,
				&pipeline_compile_options,
				&pipeline_link_options,
				program_groups,
				sizeof(program_groups) / sizeof(program_groups[0]),
				log,
				&sizeof_log,
				&pipeline
			));

			OptixStackSizes stack_sizes = {};
			for (auto& prog_group : program_groups)
			{
#if OPTIX_VERSION < 70700
				OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
#else
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
#endif
			}

			uint32_t direct_callable_stack_size_from_traversal;
			uint32_t direct_callable_stack_size_from_state;
			uint32_t continuation_stack_size;
			OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
				0,  // maxCCDepth
				0,  // maxDCDEpth
				&direct_callable_stack_size_from_traversal,
				&direct_callable_stack_size_from_state, &continuation_stack_size));
			OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
				direct_callable_stack_size_from_state, continuation_stack_size,
				max_traversal_depth  // maxTraversableDepth
			));
		}	
    }


    void createSBT(optixEnvmapVisibilityState & state) {

		OptixProgramGroup & raygen_prog_group = state.raygen_prog_group;
		OptixProgramGroup & hitgroup_prog_group = state.hitgroup_prog_group;
		OptixProgramGroup & miss_prog_group = state.miss_prog_group;

		OptixShaderBindingTable sbt = {};
		{
			CUdeviceptr  raygen_record;
			const size_t raygen_record_size = sizeof(RayGenSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
			RayGenSbtRecord rg_sbt;
			OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(raygen_record),
				&rg_sbt,
				raygen_record_size,
				cudaMemcpyHostToDevice
			));

			CUdeviceptr miss_record;
			size_t      miss_record_size = sizeof(MissSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
			MissSbtRecord ms_sbt;
			//ms_sbt.data = { 0.3f, 0.1f, 0.2f };
			OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(miss_record),
				&ms_sbt,
				miss_record_size,
				cudaMemcpyHostToDevice
			));

			CUdeviceptr hitgroup_record;
			size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
			HitGroupSbtRecord hg_sbt;
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(hitgroup_record),
				&hg_sbt,
				hitgroup_record_size,
				cudaMemcpyHostToDevice
			));

			sbt.raygenRecord = raygen_record;
			sbt.missRecordBase = miss_record;
			sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
			sbt.missRecordCount = 1;
			sbt.hitgroupRecordBase = hitgroup_record;
			sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
			sbt.hitgroupRecordCount = 1;
		}
		state.sbt = sbt; // copy OK? as only pointer and int are passed ...
	}

    void initLaunchParams(int shader, int N, int envmap_width, int envmap_height, const float * rotray, optixEnvmapVisibilityState & state )
	{
		CUstream & stream = state.stream;
		if (stream == 0) {
			CUDA_CHECK(cudaStreamCreate(&stream));
		}

		// Fill Params
		//state.params.frame_buffer = reinterpret_cast<float*>(m_buffer.d_buffer);
		state.params.handle = state.mesh->gas_handle;
		if(shader == 0){
			state.params.width = N;
			state.params.height = 1;
		}
		else if(shader == 2){
			unsigned int envmap_elem = envmap_width * envmap_height;
			state.params.width = N;
			state.params.height = (envmap_elem + 7) / 8;
		}
		else{ // 1, 3
			unsigned int envmap_elem = envmap_width * envmap_height;
			state.params.width = N;
			state.params.height = (envmap_elem + 31) / 32;
		}
        state.params.envmap_width = envmap_width;
        state.params.envmap_height = envmap_height;
		if(rotray){
			for(int i = 0; i < 9; i++){
				state.params.rotray[i] = rotray[i];
			}
		}

		optixEnvmapVisibility::LaunchParams ** d_params = &state.d_params;
		if (*d_params == 0) {
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(d_params), sizeof(optixEnvmapVisibility::LaunchParams)
			));
		}
	}

    void launchParams(
        torch::Tensor & output, torch::Tensor & rays_o, at::optional<torch::Tensor> & rays_d,
		const float * d_rotray,
        optixEnvmapVisibilityState & state, bool sync
	) {
		state.params.rays_o = reinterpret_cast<float3*>(rays_o.data_ptr());
		if(rays_d.has_value()){
			state.params.rays_d = reinterpret_cast<float3*>(rays_d.value().data_ptr());
		}
		//state.params.rays_d = reinterpret_cast<float3*>(rays_d.data_ptr());
		state.params.frame_buffer = (unsigned int *)output.data_ptr<int>();

		CUDA_CHECK(cudaMemcpyAsync(
			reinterpret_cast<void*>(state.d_params),
			&state.params, sizeof(optixEnvmapVisibility::LaunchParams),
			cudaMemcpyHostToDevice, state.stream
		));

		if(d_rotray){
			CUDA_CHECK(cudaMemcpyAsync(
				reinterpret_cast<void*>(state.d_params->rotray),
				d_rotray, sizeof(float) * 9,
				cudaMemcpyDeviceToDevice, state.stream
			));
		}

		OPTIX_CHECK(optixLaunch(
			state.pipeline,
			state.stream,
			reinterpret_cast<CUdeviceptr>(state.d_params),
			sizeof(optixEnvmapVisibility::LaunchParams),
			&state.sbt,
			state.params.width,   // launch width
			state.params.height,   // launch height
			1                     // launch depth
		));
		if(sync){
			CUDA_SYNC_CHECK();
		}
	}

	void buildMeshAccelAdv(optixEnvmapVisibilityState & state,
		std::vector<torch::Tensor> & vertex_group,
		std::vector<at::optional<at::Tensor>> & index_group,
		bool allow_update, bool update
	){
		if (state.mesh == nullptr) {
			state.mesh = new MeshInfo();
		}
		MeshInfo & mesh = *(state.mesh);
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		if(allow_update) 
			accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
		if(allow_update && update){
			building_container.Update(vertex_group, index_group);
			accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
		}
		else{
			building_container.Init(vertex_group, index_group);
			OptixAccelBufferSizes & gas_buffer_sizes = bak_buffer_size;

			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				state.context, &accel_options, 
				building_container.buildInputs.data(),
				static_cast<unsigned int>(building_container.num_subMeshes), 
				&gas_buffer_sizes));
			
			buffer_tmp_gas.allocate(gas_buffer_sizes.tempSizeInBytes);
			buffer_gas.allocate(gas_buffer_sizes.outputSizeInBytes);
		}

		// m_context -> state.context
		OPTIX_CHECK(optixAccelBuild(
			state.context,
			state.stream,         // CUDA stream
			&accel_options,
			building_container.buildInputs.data(),//&triangle_input,
			static_cast<unsigned int>(building_container.num_subMeshes), // num build inputs
			buffer_tmp_gas.cu_data,
			bak_buffer_size.tempSizeInBytes,
			buffer_gas.cu_data,
			bak_buffer_size.outputSizeInBytes,
			&mesh.gas_handle,        //&gas_info.gas_handle,
			nullptr,                    // emitted property list
			0                         // num emitted properties
		));		

		mesh.d_gas_output = buffer_gas.cu_data;
	}

    void buildMeshAccel(optixEnvmapVisibilityState & state,
		std::vector<torch::Tensor> & vertex_group,
		std::vector<at::optional<at::Tensor>> & index_group
	) {		
		if (state.mesh == nullptr) {
			state.mesh = new MeshInfo();
		}
		//loadMesh(state.mesh);
		MeshInfo & mesh = *(state.mesh);

		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		//accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		const int num_subMeshes = vertex_group.size();
		//const int num_subMeshes = 1;
		unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;
		std::vector<OptixBuildInput> buildInputs(num_subMeshes);
		std::vector<CUdeviceptr> pointers(num_subMeshes);

		for (size_t i_subindex = 0; i_subindex < num_subMeshes; i_subindex++) {

			OptixBuildInput & triangle_input = buildInputs[i_subindex];
			memset(&triangle_input, 0, sizeof(OptixBuildInput));

			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			// vertices are assumed to be tightly packed and stride is inferred from vertexFormat.
			triangle_input.triangleArray.vertexStrideInBytes = 0;
			//triangle_input.triangleArray.vertexBuffers = &(mesh.cu_vertex);
			//triangle_input.triangleArray.numVertices = mesh.n_vertex;
			
			torch::Tensor & loc_vertex = vertex_group[i_subindex];
			at::optional<at::Tensor> & loc_index = index_group[i_subindex];
			
			pointers[i_subindex] = reinterpret_cast<CUdeviceptr>(loc_vertex.data_ptr());
			triangle_input.triangleArray.vertexBuffers = &pointers[i_subindex];
			triangle_input.triangleArray.numVertices = loc_vertex.size(0);

			if(loc_index.has_value()){
				//if (mesh.flatten == 0) {
				triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				//triangle_input.triangleArray.numIndexTriplets = mesh.n_face;
				triangle_input.triangleArray.numIndexTriplets = loc_index.value().size(0);
				triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(loc_index.value().data_ptr());
			}
			else {
#if OPTIX_VERSION >= 70100
				triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
#endif
				triangle_input.triangleArray.numIndexTriplets = 0;
				triangle_input.triangleArray.indexBuffer = 0;
			}
			triangle_input.triangleArray.flags = &triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;
		}

		OptixAccelBufferSizes gas_buffer_sizes;
		// m_context -> state.context
		OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, buildInputs.data(),
			static_cast<unsigned int>(num_subMeshes), &gas_buffer_sizes));

		CUdeviceptr d_temp_buffer_gas;
		CUdeviceptr d_temp_gas_output_buffer;
		CUdeviceptr d_gas_output_buffer;

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_buffer_gas),
			gas_buffer_sizes.tempSizeInBytes
		));
		// non-compacted output	
		size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_temp_gas_output_buffer),
			compactedSizeOffset + 8
		));

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)d_temp_gas_output_buffer + compactedSizeOffset);

		// m_context -> state.context
		OPTIX_CHECK(optixAccelBuild(
			state.context,
			0,                  // CUDA stream
			&accel_options,
			buildInputs.data(),//&triangle_input,
			static_cast<unsigned int>(num_subMeshes),                  // num build inputs
			d_temp_buffer_gas,
			gas_buffer_sizes.tempSizeInBytes,
			d_temp_gas_output_buffer,
			gas_buffer_sizes.outputSizeInBytes,
			&mesh.gas_handle,        //&gas_info.gas_handle,
			&emitProperty,            // emitted property list
			1                   // num emitted properties
		));

		// We can now free the scratch space buffer used during build and the vertex
		// inputs, since they are not needed by our trivial shading method
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
		//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));

		size_t compacted_gas_size;
		CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

		if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

			// use handle as input and output
			// m_context -> state.context, gas_info.gas_handle -> mesh.gas_handle
			OPTIX_CHECK(optixAccelCompact(state.context, 0, mesh.gas_handle, d_gas_output_buffer, compacted_gas_size, &(mesh.gas_handle)));

			CUDA_CHECK(cudaFree((void*)d_temp_gas_output_buffer));

			//gas_info.d_gas_output = d_gas_output_buffer;
			mesh.d_gas_output = d_gas_output_buffer;
		}
		else
		{
			//gas_info.d_gas_output = d_temp_gas_output_buffer;
			mesh.d_gas_output = d_temp_gas_output_buffer;
		}	
	}



	void clearMesh(optixEnvmapVisibilityState & state) {
		MeshInfo * mesh = state.mesh;
		if (mesh && mesh->cu_vertex) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mesh->cu_vertex)));
			mesh->cu_vertex = 0;
		}
		if (mesh && mesh->cu_index) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mesh->cu_index)));
			mesh->cu_index = 0;
		}
		if (mesh && mesh->d_gas_output) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mesh->d_gas_output)));
			mesh->d_gas_output = 0;
		}
	}
	
	void createStream(optixEnvmapVisibilityState & state){
		CUstream& stream = state.stream;
		if (stream == 0){
			CUDA_CHECK(cudaStreamCreate(&stream));
		}
	}

	void cleanUp(optixEnvmapVisibilityState & state) {

		// delete frame
		//if (m_buffer.d_buffer) {
		//	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_buffer.d_buffer)));
		//	m_buffer.d_buffer = 0;
		//}
		// release stream
		if (state.stream) {
			CUDA_CHECK(cudaStreamDestroy(state.stream));
			state.stream = 0;
		}
		// release d_params
		if (state.d_params) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
		}

		// cuda Free SBT
		OptixShaderBindingTable & sbt = state.sbt;
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
		sbt.raygenRecord = 0;
		sbt.missRecordBase = 0;
		sbt.hitgroupRecordBase = 0;

		// clean program, module, pipeline and exit context
		OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
		OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
		OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
		OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
		OPTIX_CHECK(optixModuleDestroy(state.module));

		OPTIX_CHECK(optixDeviceContextDestroy(state.context));

		state.pipeline = 0;
		state.hitgroup_prog_group = 0;
		state.miss_prog_group = 0;
		state.module = 0;
		state.context = 0;

		if (state.mesh) {
			delete state.mesh;
			state.mesh = 0;
		}
	}

}


void optixEnvmapVisibility_init(int shader, int fullupdate_step) {
	
	// create context
	optixEnvmapVisibility::m_optixEnvmapVisibilityState.context = optixEnvmapVisibility::OptixObjCreateContext();
	optixEnvmapVisibility::createProgramGroupsAndPipeline(optixEnvmapVisibility::m_optixEnvmapVisibilityState, shader);
	optix_envmap_visibility_shader = shader;
	optix_envmap_visibility_fullupdate_step = fullupdate_step;
	optixEnvmapVisibility::createSBT(optixEnvmapVisibility::m_optixEnvmapVisibilityState);
	optixEnvmapVisibility::createStream(optixEnvmapVisibility::m_optixEnvmapVisibilityState);

}

void optixEnvmapVisibility_load_mesh(
	std::vector<torch::Tensor> vertex_group,
	std::vector<at::optional<at::Tensor>> index_group 
) {
	//optixEnvmapVisibility::clearMesh(optixEnvmapVisibility::m_optixEnvmapVisibilityState);
	//optixEnvmapVisibility::buildMeshAccel(
	//	optixEnvmapVisibility::m_optixEnvmapVisibilityState,
	//	vertex_group,
	//	index_group
	//);

	///////////////////////////////////////////////

	static int l_step = 0;
	bool allow_update = true;
	bool update = true;
	if(optix_envmap_visibility_fullupdate_step <= 0){
		allow_update = false;
	}
	else{
		if (l_step % optix_envmap_visibility_fullupdate_step == 0){
			update = false;
		}
	}

	optixEnvmapVisibility::buildMeshAccelAdv(
		optixEnvmapVisibility::m_optixEnvmapVisibilityState,
		vertex_group, index_group,
		allow_update, update
	);

	l_step++;
}

torch::Tensor optixEnvmapVisibility_trace(
    torch::Tensor rays_o, at::optional<at::Tensor> rays_d,
	torch::Tensor rot_ray, // CPU & GPU tensor
    unsigned int envmap_width, unsigned int envmap_height
){
    // rays_o Nx3
    unsigned int N = rays_o.size(0);
	bool is_cpu;
	{
		auto device = rot_ray.device();
		is_cpu = device.is_cpu();	
	}
	
	const float * rot_ray_cpu_ptr = nullptr;
	const float * rot_ray_cuda_ptr = nullptr;
	if(is_cpu){
		rot_ray_cpu_ptr = rot_ray.data_ptr<float>();
	}else{
		rot_ray_cuda_ptr = rot_ray.data_ptr<float>(); 
	}
    optixEnvmapVisibility::initLaunchParams(
        optix_envmap_visibility_shader, N, envmap_width, envmap_height, 
        rot_ray_cpu_ptr, optixEnvmapVisibility::m_optixEnvmapVisibilityState
    );

    unsigned int n_slot = (envmap_width * envmap_height + 31) / 32;
    auto device = rays_o.device();
    at::TensorOptions opt_i(at::kInt); opt_i = opt_i.device(device); opt_i = opt_i.requires_grad(false);
	torch::Tensor result;
	if(optix_envmap_visibility_shader != 2){
    	result = torch::empty({n_slot, N}, opt_i);
	}else{
		result = torch::zeros({n_slot, N}, opt_i);
	}
    //torch::Tensor result = torch::full({n_slot, N},(int)0xAAAAAAAA, opt_i);

    optixEnvmapVisibility::launchParams(
		result, rays_o, rays_d, rot_ray_cuda_ptr, 
		optixEnvmapVisibility::m_optixEnvmapVisibilityState, false
	);
    return result;
}

void optixEnvmapVisibility_destroy_mesh(){
    //optixEnvmapVisibility::clearMesh(optixEnvmapVisibility::m_optixEnvmapVisibilityState);
	buffer_gas.Destroy();
	buffer_tmp_gas.Destroy();
}

void optixEnvmapVisibility_destroy(){
    optixEnvmapVisibility::cleanUp(optixEnvmapVisibility::m_optixEnvmapVisibilityState);
}




