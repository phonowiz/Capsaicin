/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#ifndef NRC_PATH_TRACING_HLSL
#define NRC_PATH_TRACING_HLSL

#include "../../ray_tracing/path_tracing_shared.h"
#include "../../ray_tracing/trace_ray.hlsl"
#include "../../ray_tracing/intersect_data.hlsl"
#include "../../components/light_builder/light_builder.hlsl"
#include "../../components/random_number_generator/random_number_generator.hlsl"
#include "../../components/stratified_sampler/stratified_sampler.hlsl"
#include "../../geometry/mis.hlsl"
#include "../../lights/light_sampling.hlsl"
#include "../../materials/material_sampling.hlsl"
#include "../../math/transform.hlsl"
#include "nrc_common.hlsl" // For structures

// Extra Buffers for NRC
RWStructuredBuffer<InferenceQuery> g_InferenceQueries_RT : register(u5);
RWStructuredBuffer<TrainingSample> g_TrainingSamples_RT : register(u6);
RWStructuredBuffer<uint> g_Counters : register(u7); // [0]=Queries, [1]=Samples

// Reuse standard path tracing functions, but overload the main trace loop.
// We need to include the core logic files but NOT 'path_tracing.hlsl' itself to avoid conflicts if we redefine functions.
// Actually, 'path_tracing.hlsl' defines tracePath.
// We will implement our own 'tracePathNRC'.

// Include standard helpers from path_tracing.hlsl by copy-paste or trickery? 
// It's safer to just re-implement the loop using the primitives available in other includes.
// But 'pathHit', 'shadePathHit' etc are useful.
// I will include path_tracing.hlsl but use the preprocessor to rename or I'll just use the functions it provides and write a new top-level function.
// 'tracePath' is the main one. I'll write 'tracePathNRC'.

// Include Light Sampler (required for sampleLightsPDF used in path_tracing.hlsl)
#include "../../components/light_sampler/light_sampler.hlsl"

// Helper to include standard RT functions (pathMiss, pathClosestHit etc)
#include "../../ray_tracing/path_tracing_rt.hlsl"
// Parameters for the loop:
// currentAreaSpread: initialized to 0.0f
// threshold: 0.01f (as suggested in the paper)

void UpdatePathSpread(
    inout float currentAreaSpread, 
    float distToNextHit, 
    float pdf, 
    float cosThetaAtNextHit)
{
    float distSq = distToNextHit * distToNextHit;
    
    // Formula 3: Increment area spread based on the current bounce's expansion
    float expansion = distSq / (4.0f * 3.14159265f * pdf * max(cosThetaAtNextHit, 1e-4f));
    currentAreaSpread += expansion;
}



template<typename RadianceT>
void tracePathNRC(RayInfo ray, inout StratifiedSampler randomStratified, inout Random randomNG,
    uint currentBounce, uint minBounces, uint maxBounces, float3 normal, float3 throughput, inout RadianceT radiance,
    bool isTrainingRay, uint2 pixelCoord)
{
    // NRC Constants
    const uint kNRCBounce = 2; // Fixed for now

    // State for training
    TrainingSample trainingSample; 
    trainingSample.pos = 0.0f.xxx;
    trainingSample.dir = 0.0f.xxx;
    trainingSample.normal = 0.0f.xxx;
    trainingSample.roughness = 0.0f;
    trainingSample.target_radiance = 0.0f.xxx;
    bool trainingVertexFound = false;
    float3 radianceAtCachePoint = 0.0f.xxx;
    float3 throughputAtCachePoint = 1.0f.xxx;

    float samplePDF = 1.0f;
    float primaryArea = 0.0f;
    float currentAreaSpread = 0.0f;
    // Standard Path Tracing Loop
    for (uint bounce = currentBounce; bounce <= maxBounces; ++bounce)
    {
        // Check for NRC Inference Termination
        if (!isTrainingRay && bounce == kNRCBounce)
        {
            // Terminate and Emit Query
            uint queryIdx;
            InterlockedAdd(g_Counters[0], 1, queryIdx); // Atomic counter
            
            if (queryIdx < 1920*1080) // Safety
            {
                InferenceQuery q;
                q.pos = ray.origin;
                q.dir = ray.direction;
                q.normal = normal; // Approx
                q.roughness = 0.5f; // Placeholder
                q.albedo = float3(1,1,1); // Used for factorization in kernel (simplified)
                q.pixel_coord = pixelCoord;
                q.throughput = throughput; // STORE THROUGHPUT
                g_InferenceQueries_RT[queryIdx] = q;
            }
            break; // Stop tracing
        }
        
        // Trace Ray
        #if USE_INLINE_RT
        ClosestRayQuery rayQuery = TraceRay<ClosestRayQuery>(ray);
        if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
        {
             // Miss
             shadePathMiss(ray, bounce, randomNG, normal, 1.0f, throughput, radiance);
             
             // If training, this is a target value!
             if (isTrainingRay && trainingVertexFound)
             {
                 // Add this radiance to the target?
                 // Training logic is complex: it needs the suffix radiance.
                 // For now, let's just assume we train on the LAST bounce?
             }
             break;
        }
        else
        {
             // Hit
             HitInfo hitData = GetHitInfoRtInlineCommitted(rayQuery);
             IntersectData iData = MakeIntersectData(hitData);
             
             // If training ray, and we are at the "Cache point" (bounce kNRCBounce), record inputs
             if (isTrainingRay && bounce == kNRCBounce)
             {
                 trainingSample.pos = iData.position;
                 trainingSample.dir = ray.direction;
                 trainingSample.normal = iData.normal;
                 trainingSample.roughness = 0.5f;
                 trainingVertexFound = true;
                 radianceAtCachePoint = radiance;
                 throughputAtCachePoint = throughput;
                 // We continue tracing to gather the "Target Radiance"
             }

             else
             {
                float distToNextHit = length(iData.position - ray.origin);
                float cosThetaAtNextHit = abs(dot(iData.normal, ray.direction));
                UpdatePathSpread(currentAreaSpread, distToNextHit, samplePDF, cosThetaAtNextHit);

                primaryArea = (bounce == currentBounce) ? currentAreaSpread: primaryArea;

                
                if (!pathHit(ray, hitData, iData, randomStratified, randomNG,
                    bounce, minBounces, maxBounces, normal, samplePDF, throughput, radiance))
                {
                    break;
                }

                if ((currentAreaSpread - primaryArea) > 0.01f * primaryArea)
                {
                    //TODO:
                    // Path is now "blurry" enough.
                    // 1. Stop tracing rays.
                    // 2. Query the Neural Radiance Cache here.
                    break;
                }
             }
        }
        #endif
    }
    
    // If we finished a training ray, emit sample
    if (isTrainingRay && trainingVertexFound)
    {
        uint sampleIdx;
        // Limit to a reasonable batch size (e.g. 4096) to prevent overflow/TDR
        // Also checks against buffer size.
        InterlockedAdd(g_Counters[1], 1, sampleIdx);
        if (sampleIdx < 4096) 
        {
             // Target radiance is the suffix: (Final - AtBounce) / ThroughputAtBounce
             trainingSample.target_radiance = (radiance - radianceAtCachePoint) / max(throughputAtCachePoint, 1e-6f);
             g_TrainingSamples_RT[sampleIdx] = trainingSample;
        }
    }
}

#endif
