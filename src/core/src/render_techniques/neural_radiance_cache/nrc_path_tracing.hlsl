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
RWStructuredBuffer<InferenceQuery> g_TrainingSamples_RT : register(u6);
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
    float expansion = sqrt(distSq / ( pdf * max(cosThetaAtNextHit, 1e-4f)));

    currentAreaSpread += expansion;
}

float getPrimaryAreaSpread(
    float distToNextHit, 
    float cosThetaAtNextHit)
{
    return distToNextHit * distToNextHit / (4.0f * 3.14159265f * max(cosThetaAtNextHit, 1e-4f));
}


template<typename RadianceT>
void tracePathNRC(RayInfo ray, inout StratifiedSampler randomStratified, inout Random randomNG,
    uint currentBounce, uint minBounces, uint maxBounces, float3 normal, float3 throughput, inout RadianceT radiance,
    bool isTrainingRay, uint2 pixelCoord)
{
    // NRC Constants
    const uint kNRCBounce = 2; // Fixed for now
    const uint kTrainingBounce = 2; // Fixed for now

    // State for training
    // State for training
    InferenceQuery trainingSample; 
    trainingSample.pos = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.dir = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.normal = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.roughness = 0.0f;
    trainingSample.target_radiance = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.albedo = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.throughput = float4(0.0f, 0.0f, 0.0f, 0.0f);
    trainingSample.pixel_coord = uint2(0, 0);
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
                q.pos = float4(ray.origin, 0.f);
                q.dir = float4(ray.direction, 0.f);
                q.normal = float4(normal, 0.f); // Approx
                q.roughness = 0.5f; // Placeholder
                q.albedo = float4(1,1,1,0); // Used for factorization in kernel (simplified)
                q.pixel_coord = pixelCoord;
                q.throughput = float4(throughput, 0.f); // STORE THROUGHPUT
                q.target_radiance = 0.0f.xxxx;
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
             
            //  // If training ray, and we are at the "Cache point" (bounce kNRCBounce), record inputs
            //  if (isTrainingRay && bounce == kNRCBounce)
            //  {
            //      trainingSample.pos = iData.position;
            //      trainingSample.dir = ray.direction;
            //      trainingSample.normal = iData.normal;
            //      trainingSample.roughness = 0.5f;
            //      trainingVertexFound = true;
            //      radianceAtCachePoint = radiance;
            //      throughputAtCachePoint = throughput;
            //      // We continue tracing to gather the "Target Radiance"
            //  }

            //  else
            {
                float distToNextHit = length(iData.position - ray.origin);
                float cosThetaAtNextHit = abs(dot(iData.normal, ray.direction));

                if(bounce == currentBounce)
                {
                    primaryArea = getPrimaryAreaSpread(distToNextHit, cosThetaAtNextHit);
                }
                else
                {
                    UpdatePathSpread(currentAreaSpread, distToNextHit, samplePDF, cosThetaAtNextHit);
                }

                if (!pathHit(ray, hitData, iData, randomStratified, randomNG,
                    bounce, minBounces, maxBounces, normal, samplePDF, throughput, radiance))
                {
                    break;
                }

                if ((currentAreaSpread * currentAreaSpread) > 0.01f * primaryArea)
                {
                    Material material = iData.material;
                    MaterialEvaluated evalMaterial = MakeMaterialEvaluated(material, iData.uv);

                    if(isTrainingRay)
                    {

                        if(!trainingVertexFound)
                        {

                            trainingSample.pos = float4(iData.position, 0.f);
                            trainingSample.dir = float4(ray.direction, 0.f);
                            trainingSample.normal = float4(iData.normal, 0.f);
                            trainingSample.roughness = 0.5f;
                            // Use dummy values or actual material props if desired for training input? 
                            // For now keeping it simple as before, but using the struct fields.
                            trainingSample.albedo = float4(0,0,0,0); 
                            trainingSample.pixel_coord = pixelCoord;
                            trainingSample.throughput = float4(throughput, 0.f);
                            trainingVertexFound = true;
                            radianceAtCachePoint = radiance;
                            throughputAtCachePoint = throughput;
                            currentAreaSpread = 0;  
                        }
                        else break;
                    }
                    else
                    {
                        uint queryIdx;
                        InterlockedAdd(g_Counters[0], 1, queryIdx);
                        if (queryIdx < 1920*1080) // Safety
                        {
                            InferenceQuery q;
                            q.pos = float4(iData.position, 0.f);
                            q.dir = float4(ray.direction, 0.f);
                            q.normal = float4(iData.normal, 0.f);
                            q.roughness = evalMaterial.roughness;
                            q.albedo = float4(evalMaterial.albedo, 0.f);
                            q.pixel_coord = pixelCoord;
                            q.throughput = float4(throughput, 0.f);
                            q.target_radiance = 0.0f.xxxx;
                            g_InferenceQueries_RT[queryIdx] = q;
                        }
                        break;
                    }
                }
            }
        }
        #endif
    }
    
    //radiance = 0.0f.xxx;
    // If we finished a training ray, emit sample
    if (isTrainingRay && trainingVertexFound)
    {

        //break;
        uint sampleIdx;
        // Limit to a reasonable batch size (e.g. 4096) to prevent overflow/TDR
        // Also checks against buffer size.
        InterlockedAdd(g_Counters[1], 1, sampleIdx);
        if (sampleIdx < 1920 * 1080) 
        {
            //radiance = float3(1.0f, 0.0f, 0.0f);
             //what we are doing here is isolating the suffix radiance from the prefix radiance.
             //We do this by subtracting the radiance at the cache point from the final radiance.
             //Then we divide by the throughput at the cache point to get the raw suffix radiance.
            trainingSample.target_radiance.xyzw = float4((radiance - radianceAtCachePoint) / max(throughputAtCachePoint, 1e-6f), 1.0f);
            g_TrainingSamples_RT[sampleIdx] = trainingSample;

            //radiance  = 0.0f.xxx;
        }
    }
}

#endif
