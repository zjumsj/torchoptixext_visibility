#include "optix.h"
#include "optix_stubs.h"
#if OPTIX_VERSION < 70300
// <= 7.2.x
#include "OptixUtil.h"
#else
#include "optix_stack_size.h"
#endif

#include <cuda_runtime.h>
#include "optixDist.h"
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

typedef SbtRecord<optixDist::RayGenData>     RayGenSbtRecord;
typedef SbtRecord<optixDist::MissData>       MissSbtRecord;
typedef SbtRecord<optixDist::HitGroupData>   HitGroupSbtRecord;

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


struct optixDistState {

	OptixDeviceContext context = 0;
	MeshInfo *         mesh = 0;

	CUstream            stream = 0;

	OptixProgramGroup   hitgroup_prog_group = 0;
	OptixProgramGroup   miss_prog_group = 0;
	OptixProgramGroup   raygen_prog_group = 0;
	OptixModule         module = 0;
	OptixPipeline       pipeline = 0;

	OptixShaderBindingTable  sbt = {};

	optixDist::LaunchParams  params;
	optixDist::LaunchParams*  d_params;
};

//struct frameBuffer {
//
//	int width = 0;
//	int height = 0;
//	CUdeviceptr  d_buffer = 0;
//
//};

// Avoid namespace collision
namespace optixDist{

	//frameBuffer m_buffer;
	optixDistState m_optixDistState;
	
	//CUdeviceptr d_in_rays_o;
	//CUdeviceptr d_in_rays_d;
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

namespace optixDist {

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

	void createProgramGroupsAndPipeline( optixDistState & state)
	{
		//const char * cu_file = "cu/optixDist.cu";

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
			//pipeline_compile_options.numPayloadValues = 3;
			//pipeline_compile_options.numAttributeValues = 3;

			pipeline_compile_options.numPayloadValues = 1;
			pipeline_compile_options.numAttributeValues = 2; // 2 for triangles

#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
			pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
#if OPTIX_VERSION >= 70100
			pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#endif
		}
		// Create Module
		OptixModule & module = state.module;
		{
			std::string ptx;
			//const std::string module_name = "cu/optixDist.cu";
			const std::string module_name = myTools::cat_path(gb_path.c_str(), "cu/optixDist.cu");
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
			raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

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

	void createSBT(optixDistState & state) {

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

	void initLaunchParams(int width, int height, optixDistState & state )
	{
		CUstream & stream = state.stream;
		if (stream == 0) {
			CUDA_CHECK(cudaStreamCreate(&stream));
		}

		// Fill Params
		//state.params.frame_buffer = reinterpret_cast<float*>(m_buffer.d_buffer);
		state.params.handle = state.mesh->gas_handle;
		state.params.width = width;
		state.params.height = height;

		optixDist::LaunchParams ** d_params = &state.d_params;
		if (*d_params == 0) {
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(d_params), sizeof(optixDist::LaunchParams)
			));
		}
	}

	void launchParams(
		torch::Tensor & output, torch::Tensor & rays_o, torch::Tensor & rays_d,
		optixDistState & state
	) {
		state.params.rays_o = reinterpret_cast<float3*>(rays_o.data_ptr());
		state.params.rays_d = reinterpret_cast<float3*>(rays_d.data_ptr());
		state.params.frame_buffer = output.data_ptr<float>();

		CUDA_CHECK(cudaMemcpyAsync(
			reinterpret_cast<void*>(state.d_params),
			&state.params, sizeof(optixDist::LaunchParams),
			cudaMemcpyHostToDevice, state.stream
		));

		OPTIX_CHECK(optixLaunch(
			state.pipeline,
			state.stream,
			reinterpret_cast<CUdeviceptr>(state.d_params),
			sizeof(optixDist::LaunchParams),
			&state.sbt,
			state.params.width,   // launch width
			state.params.height,  // launch height
			1                     // launch depth
		));
		CUDA_SYNC_CHECK();
	}

	void buildMeshAccel(optixDistState & state,
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

	void clearMesh(optixDistState & state) {
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
	
	void cleanUp(optixDistState & state) {

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

void optixDist_init() {
	
	// create context
	optixDist::m_optixDistState.context = optixDist::OptixObjCreateContext();
	optixDist::createProgramGroupsAndPipeline(optixDist::m_optixDistState);
	optixDist::createSBT(optixDist::m_optixDistState);
}

void optixDist_load_mesh(
	std::vector<torch::Tensor> vertex_group,
	std::vector<at::optional<at::Tensor>> index_group 
) {
	optixDist::clearMesh(optixDist::m_optixDistState);
	optixDist::buildMeshAccel(
		optixDist::m_optixDistState,
		vertex_group,
		index_group
	);
}

torch::Tensor optixDist_trace(
	torch::Tensor rays_o,
	torch::Tensor rays_d
) {
	// rays_o HxWx3
	// rays_d HxWx3
	int H = rays_o.size(0);
	int W = rays_o.size(1);
	optixDist::initLaunchParams(W, H, optixDist::m_optixDistState);
	
	//optixDist::initLaunchParams(m_optixDistState);
	auto device = rays_o.device();
	at::TensorOptions opt(rays_o.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
	torch::Tensor result = torch::empty({ H,W }, opt);

	optixDist::launchParams(result, rays_o, rays_d, optixDist::m_optixDistState);
	return result;
}

void optixDist_destroy_mesh() {
	optixDist::clearMesh(optixDist::m_optixDistState);
}

void optixDist_destroy() {
	// clear rays_o, rays_d ?
	// clear mesh
	optixDist::cleanUp(optixDist::m_optixDistState);
}

