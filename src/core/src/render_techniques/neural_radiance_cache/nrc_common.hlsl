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
    uint activations_stride;
    uint activations_offset;
    uint is_training_pass;
};

// Inference Inputs: [Pos (3), Dir (3), Supplemental (Normal, Roughness?)] -> Encoded to 64
// Inference Inputs: [Pos (3), Dir (3), Supplemental (Normal, Roughness?)] -> Encoded to 64
struct InferenceQuery
{
    float4 pos;
    float4 dir;
    float4 normal;
    float4 albedo;
    float4 throughput;
    float4 target_radiance;
    uint2  pixel_coord;
    float  roughness;
};

// --- Helper Functions ---
// Helper for Frequency Encoding using Triangle Waves
float triangle_wave(float x) {
    return abs(frac(x + 0.5) * 2.0 - 1.0);
}

// Helper for One-Blob Encoding using Quartic Kernels
float quartic_kernel(float x) {
    float v = 1.0 - x * x;
    return (x >= -1.0 && x <= 1.0) ? (15.0/16.0) * v * v : 0.0;
}

void EncodeInputs(InferenceQuery query, out float features[NETWORK_WIDTH])
{
    float3 pos = query.pos.xyz;
    float3 viewDir = query.dir.xyz;
    float3 normal = query.normal.xyz;
    float roughness = query.roughness;
    float3 diffuse = query.albedo.rgb;

    //todo: add specular
    float3 specular = 0;//query.albedo.rgb;
    int idx = 0;

    // 1. Frequency Encoding for Position (3 components * 12 frequencies = 36 dims)
    for (int i = 0; i < 3; ++i) {
        float val = pos[i];
        for (int f = 0; f < 12; ++f) {
            features[idx++] = triangle_wave(val * pow(2.0, f));
        }
    }

    // 2. One-Blob Encoding for Direction & Normals (2 params * 2D * 4 blobs = 16 dims)
    float2 dir_sph = float2(acos(viewDir.y), atan2(viewDir.z, viewDir.x));
    float2 norm_sph = float2(acos(normal.y), atan2(normal.z, normal.x));
    float2 sph_coords[2] = { dir_sph, norm_sph };

    for (int p = 0; p < 2; ++p) {
        for (int d = 0; d < 2; ++d) {
            float val = sph_coords[p][d];
            for (int b = 0; b < 4; ++b) {
                features[idx++] = quartic_kernel(val - (float(b) / 3.0));
            }
        }
    }

    // 3. One-Blob Encoding for Roughness (1 param * 4 blobs = 4 dims)
    float r_mapped = 1.0 - exp(-roughness);
    for (int b = 0; b < 4; ++b) {
        features[idx++] = quartic_kernel(r_mapped - (float(b) / 3.0));
    }

    // 4. Identity Encoding for Reflectance (3 Diffuse + 3 Specular = 6 dims)
    features[idx++] = diffuse.r;
    features[idx++] = diffuse.g;
    features[idx++] = diffuse.b;
    features[idx++] = specular.r;
    features[idx++] = specular.g;
    features[idx++] = specular.b;

    // 5. Padding (Total so far: 62. Pad to 64)
    features[idx++] = 1.0; 
    features[idx++] = 1.0;
}

#endif
