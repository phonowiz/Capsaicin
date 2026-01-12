/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#ifndef NRC_COMMON_HLSL
#define NRC_COMMON_HLSL

#define GROUP_SIZE 128
#define NETWORK_WIDTH 64
#define HIDDEN_LAYERS 5

// Use float for now for stability and compatibility
typedef float NrcFloat;
typedef float2 NrcFloat2;
typedef float3 NrcFloat3;
typedef float4 NrcFloat4;

struct NRCConstants
{
    uint num_training_samples;
    uint num_inference_queries;
    float learning_rate;
    uint batch_size;
};

// Inference Inputs: [Pos (3), Dir (3), Supplemental (Normal, Roughness?)] -> Encoded to 64
struct InferenceQuery
{
    float3 pos;
    float3 dir;
    float3 normal;
    float roughness;
    float3 albedo;
    uint2 pixel_coord;
    float3 throughput;
};

struct TrainingSample
{
    float3 pos;
    float3 dir;
    float3 normal;
    float roughness;
    float3 target_radiance;
};

// --- Helper Functions ---
void EncodeInput(InferenceQuery query, out NrcFloat features[NETWORK_WIDTH])
{
    // Zero out
    for(int i=0; i<NETWORK_WIDTH; ++i) features[i] = 0.0f;

    int idx = 0;
    float3 p = query.pos;
    float3 d = query.dir;
    float3 n = query.normal;
    
    // Position frequencies
    float freqs[] = { 1.0f, 2.0f, 4.0f, 8.0f };
    [unroll]
    for(int f=0; f<4; ++f)
    {
        float3 val = p * freqs[f];
        features[idx++] = sin(val.x); features[idx++] = cos(val.x);
        features[idx++] = sin(val.y); features[idx++] = cos(val.y);
        features[idx++] = sin(val.z); features[idx++] = cos(val.z);
    }

    // Dir
    [unroll]
    for(int f=0; f<2; ++f)
    {
        float3 val = d * freqs[f];
        features[idx++] = sin(val.x); features[idx++] = cos(val.x);
        features[idx++] = sin(val.y); features[idx++] = cos(val.y);
        features[idx++] = sin(val.z); features[idx++] = cos(val.z);
    }
    
    // Normal
    [unroll]
    for(int f=0; f<2; ++f)
    {
        float3 val = n * freqs[f];
        features[idx++] = sin(val.x); features[idx++] = cos(val.x);
        features[idx++] = sin(val.y); features[idx++] = cos(val.y);
        features[idx++] = sin(val.z); features[idx++] = cos(val.z);
    }
    
    features[idx++] = query.roughness;
    features[idx++] = sin(query.roughness * 3.14159f);
    features[idx++] = cos(query.roughness * 3.14159f);
}

void EncodeInputTraining(TrainingSample sample, out NrcFloat features[NETWORK_WIDTH])
{
    InferenceQuery q;
    q.pos = sample.pos;
    q.dir = sample.dir;
    q.normal = sample.normal;
    q.roughness = sample.roughness;
    q.albedo = float3(0,0,0);
    q.pixel_coord = uint2(0,0);
    q.throughput = float3(0,0,0);
    EncodeInput(q, features);
}

#endif
