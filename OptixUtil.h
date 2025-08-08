#pragma once
#ifndef OPTIXUTIL_H
#define OPTIXUTIL_H

#include "optix.h"

#ifdef __cplusplus
extern "C" {
#endif

	OptixResult optixUtilAccumulateStackSizes(OptixProgramGroup programGroup, OptixStackSizes* stackSizes);

	OptixResult optixUtilComputeStackSizes(const OptixStackSizes* stackSizes,
		unsigned int           maxTraceDepth,
		unsigned int           maxCCDepth,
		unsigned int           maxDCDepth,
		unsigned int*          directCallableStackSizeFromTraversal,
		unsigned int*          directCallableStackSizeFromState,
		unsigned int*          continuationStackSize
	);

	OptixResult optixUtilComputeStackSizesDCSplit(const OptixStackSizes* stackSizes,
		unsigned int           dssDCFromTraversal,
		unsigned int           dssDCFromState,
		unsigned int           maxTraceDepth,
		unsigned int           maxCCDepth,
		unsigned int           maxDCDepthFromTraversal,
		unsigned int           maxDCDepthFromState,
		unsigned int*          directCallableStackSizeFromTraversal,
		unsigned int*          directCallableStackSizeFromState,
		unsigned int*          continuationStackSize
	);

	OptixResult optixUtilComputeStackSizesCssCCTree(const OptixStackSizes* stackSizes,
		unsigned int           cssCCTree,
		unsigned int           maxTraceDepth,
		unsigned int           maxDCDepth,
		unsigned int*          directCallableStackSizeFromTraversal,
		unsigned int*          directCallableStackSizeFromState,
		unsigned int*          continuationStackSize
	);

	OptixResult optixUtilComputeStackSizesDCSplit(const OptixStackSizes* stackSizes,
		unsigned int           dssDCFromTraversal,
		unsigned int           dssDCFromState,
		unsigned int           maxTraceDepth,
		unsigned int           maxCCDepth,
		unsigned int           maxDCDepthFromTraversal,
		unsigned int           maxDCDepthFromState,
		unsigned int*          directCallableStackSizeFromTraversal,
		unsigned int*          directCallableStackSizeFromState,
		unsigned int*          continuationStackSize
	);

	OptixResult optixUtilComputeStackSizesSimplePathTracer(OptixProgramGroup        programGroupRG,
		OptixProgramGroup        programGroupMS1,
		const OptixProgramGroup* programGroupCH1,
		unsigned int             programGroupCH1Count,
		OptixProgramGroup        programGroupMS2,
		const OptixProgramGroup* programGroupCH2,
		unsigned int             programGroupCH2Count,
		unsigned int*            directCallableStackSizeFromTraversal,
		unsigned int*            directCallableStackSizeFromState,
		unsigned int*            continuationStackSize
	);

#ifdef __cplusplus
}
#endif

#endif
