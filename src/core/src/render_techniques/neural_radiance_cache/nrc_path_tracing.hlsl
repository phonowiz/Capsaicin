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



#define MAX_TRAINING_VERTICES 12

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
    uint trainingVertexCount = 0;
    InferenceQuery trainingVertices[MAX_TRAINING_VERTICES];


    float samplePDF = 1.0f;
    float primaryArea = 0.0f;
    float currentAreaSpread = 0.0f;

    // enum TerminateType
    // {
    //     kMiss,
    //     kMaxBounces,
    //     kThreshold
    // };
    //TerminateType terminateType = kMiss;

    InferenceQuery q;
    q.pixel_coord = -1;
    // Standard Path Tracing Loop
    for (uint bounce = currentBounce; bounce <= maxBounces; ++bounce)
    {
        // Trace Ray
        #if USE_INLINE_RT
        ClosestRayQuery rayQuery = TraceRay<ClosestRayQuery>(ray);
        if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
        {
            // Miss
            shadePathMiss(ray, bounce, randomNG, normal, 1.0f, throughput, radiance);
            //terminateType = kMiss;


            if (!isTrainingRay && any(q.pixel_coord != -1))
            {
                // Terminate and Emit Query
                uint queryIdx;
                InterlockedAdd(g_Counters[0], 1, queryIdx); 
                
                if (queryIdx < 1920*1080) 
                {
                    g_InferenceQueries_RT[queryIdx] = q;
                }
            }

            break;
        }
        else
        {
            // Hit
            HitInfo hitData = GetHitInfoRtInlineCommitted(rayQuery);
            IntersectData iData = MakeIntersectData(hitData);

            Material material = iData.material;
            MaterialEvaluated evalMaterial = MakeMaterialEvaluated(material, iData.uv);

            q.pos = float4(iData.position, 0.f);
            q.dir = float4(ray.direction, 0.f);
            q.normal = float4(iData.normal, 0.f);
            q.roughness = evalMaterial.roughness;
            q.albedo = float4(evalMaterial.albedo, 1.f); 
            q.pixel_coord = pixelCoord;
            q.throughput = float4(throughput, 0.f);
            // Store current accumulated radiance as temporary prefix
            q.target_radiance = float4(radiance, 1.f);
             
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
                    if (!isTrainingRay && any(q.pixel_coord != -1))
                    {
                        // Terminate and Emit Query
                        uint queryIdx;
                        InterlockedAdd(g_Counters[0], 1, queryIdx); 
                        
                        if (queryIdx < 1920*1080) 
                        {
                            g_InferenceQueries_RT[queryIdx] = q;
                        }
                    }
                    break;
                }
                
                if ((currentAreaSpread * currentAreaSpread) > 0.01f * primaryArea || bounce == maxBounces)
                {
                    //terminateType = kThreshold;
                    // Check for NRC Inference Termination (moved to Hit)
                    if (!isTrainingRay && (bounce == kNRCBounce || bounce == maxBounces))
                    {
                        // Terminate and Emit Query
                        uint queryIdx;
                        InterlockedAdd(g_Counters[0], 1, queryIdx); 
                        
                        if (queryIdx < 1920*1080) 
                        {
                            g_InferenceQueries_RT[queryIdx] = q;
                        }
                        break; // Stop tracing
                    }

                    if(isTrainingRay)
                    {
                        if(trainingVertexCount < MAX_TRAINING_VERTICES)
                        {
                            trainingVertices[trainingVertexCount] = q;
                            trainingVertexCount++;
                            
                            if(trainingVertexCount == 2)
                                currentAreaSpread = 0;  // reset spread for tail section of path trace
                        }
                    }
                    else
                    {
                        uint queryIdx;
                        InterlockedAdd(g_Counters[0], 1, queryIdx);
                        if (queryIdx < 1920*1080) // Safety
                        {
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
    // Emit all training samples
    if (isTrainingRay && trainingVertexCount > 0)
    {
        for (uint i = 0; i < trainingVertexCount; ++i)
        {
            uint sampleIdx;
            InterlockedAdd(g_Counters[1], 1, sampleIdx);
            
            if (sampleIdx < 20736) 
            {
                // Retrieve stored prefix radiance and throughput
                float3 prefixRadiance = trainingVertices[i].target_radiance.xyz;
                float3 vertexThroughput = trainingVertices[i].throughput.xyz;

                // Calculate suffix radiance: (Final - Prefix) / Throughput
                float3 suffixRadiance = (radiance - prefixRadiance) / max(vertexThroughput, 1e-6f);
                
                trainingVertices[i].target_radiance = float4(suffixRadiance, 1.0f);
                g_TrainingSamples_RT[sampleIdx] = trainingVertices[i];
            }
        }
    }
}

#endif
