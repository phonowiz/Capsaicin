/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#include "nrc_common.hlsl"

// #include "../../gpu_shared.h" // Removed to avoid include path issues. 
// Standard HLSL types used.

#define GROUP_SIZE 128
#define NETWORK_WIDTH 64
#define HIDDEN_LAYERS 6


// Use 16-bit types for performance (Requires -enable-16bit-types and SM 6.2+)
typedef half  float16_t;
typedef half2 float16_t2;
typedef half3 float16_t3;
typedef half4 float16_t4;

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

ConstantBuffer<NRCConstants> g_NRCConstants : register(b0);

// Weights buffer: [Layer 0 (64x64 + 64 bias) | Layer 1 | ... ]
// For simplicity, let's assume fully connected layers without explicit bias in the matrix for now (or bias concatenated)
// The reference kernel uses 64 inputs -> 64 outputs.
//RWStructuredBuffer<float16_t2> g_Weights : register(u0);

// Inference Inputs: [Pos (3), Dir (3), Supplemental (Normal, Roughness?)] -> Encoded to 64
// We will assume the input buffer already contains encoded features for this pass, 
// or we do on-the-fly encoding. Ideally on-the-fly.
// Let's store raw queries and encode in kernel.
// struct InferenceQuery
// {
//     float3 pos;
//     float3 dir;
//     float3 normal;
//     float roughness;
//     float3 albedo;
//     uint2 pixel_coord;
//     float3 throughput; // Added
// };

StructuredBuffer<InferenceQuery> g_InferenceQueries : register(t0);
RWTexture2D<float4> g_OutputTexture : register(u1);
RWTexture2D<float4> g_AccumulationBuffer : register(u5); // Bind Acc Buffer here


struct TrainingSample
{
    float3 pos;
    float3 dir;
    float3 normal;
    float roughness;
    float3 target_radiance;
};

StructuredBuffer<TrainingSample> g_TrainingSamples : register(t1);

// Gradient accumulation buffer
RWStructuredBuffer<NrcFloat> g_Gradients : register(u2);

// Momentum buffers for Adam/SGD
RWStructuredBuffer<NrcFloat> g_Momentum1 : register(u3);
RWStructuredBuffer<NrcFloat> g_Momentum2 : register(u4);

StructuredBuffer<uint> g_Counters : register(t2); // [0]=InferenceCount, [1]=TrainCount


// Redundant encoding helpers removed (now in nrc_common.hlsl)


void EncodeInputTraining(TrainingSample sample, out NrcFloat features[NETWORK_WIDTH])
{
    InferenceQuery q;
    q.pos = sample.pos;
    q.dir = sample.dir;
    q.normal = sample.normal;
    q.roughness = sample.roughness;
    EncodeInputs(q, features);
}


/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

// /**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#include "nrc_common.hlsl"

// Enable 16-bit types
typedef half  float16_t;
typedef half2 float16_t2;

#define NETWORK_WIDTH 64
#define LAYER_WIDTH_HALVES 32   
#define THREAD_NEURONS_HALVES 8   
#define BLOCK_PIXELS 128        

ConstantBuffer<NRCConstants> g_NRCConstants : register(b0);

StructuredBuffer<float16_t2> g_Weights : register(t1);
StructuredBuffer<InferenceQuery> g_InferenceQueries : register(t0);
StructuredBuffer<uint> g_Counters : register(t2);

// Output: intermediate activations for backward pass
// Size = [NumLayers (7)][NumQueries][32 half2]
RWStructuredBuffer<float16_t2> g_Activations : register(u2);

RWTexture2D<float4> g_OutputTexture : register(u1);
RWTexture2D<float4> g_AccumulationBuffer : register(u5);

groupshared float16_t2 s_activations[LAYER_WIDTH_HALVES][BLOCK_PIXELS];
groupshared float16_t2 s_weights[LAYER_WIDTH_HALVES * 64];

[numthreads(BLOCK_PIXELS, 1, 1)]
void NRCInference(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint pixel_in_block = gtid.x;
    const uint warp_id = gtid.x / 32;
    const uint global_pixel_idx = dtid.x;

    uint count = g_Counters[0];
    if (global_pixel_idx >= count)
        return;

    InferenceQuery query = g_InferenceQueries[global_pixel_idx];
    
    // 1. Encode Input
    NrcFloat activations_f[NETWORK_WIDTH];
    EncodeInputs(query, activations_f);
    
    // Store initial encoded features into Shared Memory
    [unroll]
    for (int i = 0; i < LAYER_WIDTH_HALVES; i++)
    {
        s_activations[i][pixel_in_block] = float16_t2(
            (float16_t)activations_f[i * 2], 
            (float16_t)activations_f[i * 2 + 1]
        );
    }

    // --- MLP Layers ---
    [loop]
    for (int layer = 0; layer < 7; layer++)
    {
        // A. LOG ACTIVATIONS (Before they are overwritten by matrix multiply)
        // These are the inputs to the current 'layer'
        [unroll]
        for (int k = 0; k < LAYER_WIDTH_HALVES; k++)
        {
            uint act_idx = (layer * g_NRCConstants.num_inference_queries + global_pixel_idx) * LAYER_WIDTH_HALVES + k;
            g_Activations[act_idx] = s_activations[k][pixel_in_block];
        }

        const int LAYER_WEIGHT_STRIDE = NETWORK_WIDTH * LAYER_WIDTH_HALVES;
        // B. Cooperative Weight Load
        [unroll]
        for (int j = 0; j < 16; j++)
        {
            int w_idx = j * BLOCK_PIXELS + pixel_in_block;
            s_weights[w_idx] = g_Weights[layer * LAYER_WEIGHT_STRIDE + w_idx];
        }
        GroupMemoryBarrierWithGroupSync();

        // C. Compute (Fused half2 Multiply-Accumulate)
        float16_t2 my_results[THREAD_NEURONS_HALVES]; 
        [unroll]
        for (int out_pair = 0; out_pair < THREAD_NEURONS_HALVES; out_pair++)
        {
            float16_t2 sum2 = (float16_t2)0.0;
            int row_index = (warp_id * THREAD_NEURONS_HALVES) + out_pair;

            [unroll]
            for (int in_pair = 0; in_pair < LAYER_WIDTH_HALVES; in_pair++)
            {
                sum2 += s_activations[in_pair][pixel_in_block] * s_weights[row_index * LAYER_WIDTH_HALVES + in_pair];
            }
            // ReLU
            my_results[out_pair] = max(sum2, (float16_t2)0.0);
        }

        // D. Hand-off (Write the new activations for the next loop/layer)
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (int out_p = 0; out_p < THREAD_NEURONS_HALVES; out_p++)
        {
            int r_idx = (warp_id * THREAD_NEURONS_HALVES) + out_p;
            s_activations[r_idx][pixel_in_block] = my_results[out_p];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Final Write-out (Reducing 64 neurons to RGB and Factorizing)
    float16_t2 rg_raw = s_activations[0][pixel_in_block];
    float16_t  b_raw  = s_activations[1][pixel_in_block].x;

    float3 radiance = float3((float)rg_raw.x, (float)rg_raw.y, (float)b_raw);
    radiance = max(0.0f, radiance);
    
    float3 final_color = radiance * (query.albedo + 0.001f);
    
    // Accumulation Buffer Logic
    float4 acc = g_AccumulationBuffer[query.pixel_coord];
    float3 current_radiance = acc.xyz;
    uint sample_count = asuint(acc.w);
    
    if (sample_count > 0)
    {
        float3 updated = current_radiance + (final_color * query.throughput / (float)sample_count);
        g_AccumulationBuffer[query.pixel_coord] = float4(updated, asfloat(sample_count));
    }
    
    g_OutputTexture[query.pixel_coord] = float4(final_color, 1.0f);
}

// Forward Pass
// Uses Shared Memory for weights to reuse across thread block.
// Groupshared: We need to load 64x64 float weights. 4KB. Fits easily.
// groupshared NrcFloat s_LayerWeights[NETWORK_WIDTH * NETWORK_WIDTH];

// [numthreads(GROUP_SIZE, 1, 1)]
// void NRCInference(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
// {
//     uint count = g_Counters[0];
//     if (dtid.x >= count) return; // Dynamic count check

//     InferenceQuery query = g_InferenceQueries[dtid.x];
    
//     // 1. Encode
//     NrcFloat activations[NETWORK_WIDTH][GROUP_SIZE];
//     //EncodeInput(query, activations);
//     EncodeInputs(query, activations);
    
//     // 2. Run Layers
//     for (int layer = 0; layer < HIDDEN_LAYERS; ++layer) // +2 for In/Out?
//     {
//         // For simplicity, let's do 5 layers of 64x64.
//         // Load Weights into Shared Memory cooperatively
//         // Total weights: 64*64 = 4096.
//         // Threads: 128. Each thread loads 32 floats.
//         uint weight_offset = layer * NETWORK_WIDTH * NETWORK_WIDTH;
        
//         for (int i = 0; i < 32; ++i)
//         {
//             int idx = i * GROUP_SIZE + gtid.x;
//             if (idx < NETWORK_WIDTH * NETWORK_WIDTH)
//             {
//                 s_LayerWeights[idx] = g_Weights[weight_offset + idx];
//             }
//         }
//         GroupMemoryBarrierWithGroupSync();
        
//         // Matrix Multiply: Output = Activation * Weight
//         // We calculate new activations for this thread
//         NrcFloat next_activations[NETWORK_WIDTH];
        
//         for (int out_row = 0; out_row < NETWORK_WIDTH; ++out_row)
//         {
//             NrcFloat sum = 0.0f;
//             for (int in_col = 0; in_col < NETWORK_WIDTH; ++in_col)
//             {
//                 // W is stored Row-Major? Let's assume W[row][col]
//                 // Output[row] = Sum(Input[col] * W[row][col])
//                 sum += activations[in_col] * s_LayerWeights[out_row * NETWORK_WIDTH + in_col];
//             }
            
//             // ReLU (except last layer?)
//             // For now, ReLU all hidden.
//             if (layer < HIDDEN_LAYERS)
//                  next_activations[out_row] = max(0.0f, sum);
//             else
//                  next_activations[out_row] = sum; // Sigmoid or Linear for last?
//         }
        
//         // Update registers
//         activations = next_activations;
        
//         GroupMemoryBarrierWithGroupSync();
//     }
    
//     // 3. Output
//     // First 3 floats are RGB.
//     // Factorize with Albedo (from query).
//     float3 radiance = float3(activations[0], activations[1], activations[2]);
    
//     // Apply albedo factorization (demodulation)
//     // In paper: Network predicts "Radiance / Albedo" (plus some factor), so we multiply back.
//     // Ensure positive
//     radiance = max(0.0f, radiance);
    
//     // Write Output
//     // Simple Exponential accumulation check
//     float3 final_color = radiance * (query.albedo + 0.001f); // Multiply by albedo
    
//     // Add to Accumulation
//     // Note: This is racy if multiple queries hit same pixel? 
//     // Usually 1 query per pixel if bounce=2.
//     // If we have jitter, we need to access the AccBuffer carefully.
//     // But let's assume standard accumulation logic happens in RayGen for primary, here for indirect.
//     // "radiance" here is L_indirect.
//     // L_total = L_primary + Throughput * L_indirect
    
//     float4 acc = g_AccumulationBuffer[query.pixel_coord];
//     float3 current_radiance = acc.xyz;
//     uint sample_count = asuint(acc.w);
    
//     // Weighted update? The RayGen already divided L_primary by N.
//     // We should divide this by N too.
//     // BUT RayGen did: rad = rad_prev + (new - rad_prev)/N.
//     // It's easier if RayGen writes (L_primary) and we ADD (Throughput * L_nrc / N).
//     // Or we just strictly add to the current frame's radiance before averaging?
//     // Let's assume RayGen wrote `Prev + (Primary - Prev)/N`.
//     // We want `Prev + ((Primary + T*NRC) - Prev)/N`.
//     // Difference is `(T*NRC)/N`.
    
//     if (sample_count > 0)
//         g_AccumulationBuffer[query.pixel_coord] = float4(current_radiance + (final_color * query.throughput / (float)sample_count), asfloat(sample_count));
    
//     g_OutputTexture[query.pixel_coord] = float4(final_color, 1.0f);
// }

// Training Kernel
// Naive implementation: One thread per sample. 
// Standard backprop.
[numthreads(GROUP_SIZE, 1, 1)]
void NRCTrain(uint3 dtid : SV_DispatchThreadID)
{
    uint count = g_Counters[1];
    if (dtid.x >= count) return;
    
    TrainingSample sample = g_TrainingSamples[dtid.x];
    
    // Forward Pass (Record activations needed for backprop)
    // We need to store activations for all layers for this sample.
    // Stack: [Layers][Width]
    NrcFloat layer_activations[HIDDEN_LAYERS + 1][NETWORK_WIDTH];
    
    EncodeInputTraining(sample, layer_activations[0]);
    
    for (int layer = 0; layer < HIDDEN_LAYERS; ++layer)
    {
         uint weight_offset = layer * NETWORK_WIDTH * NETWORK_WIDTH;
         
         for (int r = 0; r < NETWORK_WIDTH; ++r)
         {
             NrcFloat sum = 0.0f;
             for (int c = 0; c < NETWORK_WIDTH; ++c)
             {
                 sum += layer_activations[layer][c] * g_Weights[weight_offset + r * NETWORK_WIDTH + c];
             }
             layer_activations[layer+1][r] = max(0.0f, sum); // ReLU
         }
    }
    
    // Output Layer (Last transform)
    // This is simplified. Normally output is smaller (3 dims).
    // Let's assume Layer 5 maps 64 -> 64, and we just take first 3.
    // Real implementation would have a specific output layer size.
    
    // Calc Loss Gradients
    float3 prediction = float3(layer_activations[HIDDEN_LAYERS][0], layer_activations[HIDDEN_LAYERS][1], layer_activations[HIDDEN_LAYERS][2]);
    float3 target = sample.target_radiance; // This should be Demodulated Radiance (Radiance / Albedo)
    
    // L2 Loss: 0.5 * (Pred - Target)^2
    // Grad: (Pred - Target)
    float3 dL_dPred = prediction - target;
    
    // Backprop
    NrcFloat dL_dX[NETWORK_WIDTH]; // Gradients for current layer neurons
    
    // Init dL_dX for output layer
    for(int i=0; i<NETWORK_WIDTH; ++i) dL_dX[i] = 0.0f;
    dL_dX[0] = dL_dPred.x;
    dL_dX[1] = dL_dPred.y;
    dL_dX[2] = dL_dPred.z;
    
    // Loop backwards
    for (int layer = HIDDEN_LAYERS - 1; layer >= 0; --layer)
    {
         uint weight_offset = layer * NETWORK_WIDTH * NETWORK_WIDTH;
         NrcFloat dL_dX_prev[NETWORK_WIDTH]; // For input of this layer
         for(int k=0; k<NETWORK_WIDTH; ++k) dL_dX_prev[k] = 0.0f;
         
         for (int r = 0; r < NETWORK_WIDTH; ++r)
         {
             // If ReLU was active
             NrcFloat output_val = layer_activations[layer+1][r];
             NrcFloat grad = (output_val > 0.0f) ? dL_dX[r] : 0.0f;
             
             // Update Weights Gradients
             // dL_dW_rc = dL_dOutput_r * Input_c
             // Atomic Add to Global Gradients? VERY SLOW.
             // Better: Compute local gradients, then do a reduction or have one thread do a batch?
             // Standard SGD is noisy.
             // For this naive implementation, we will perform atomic adds.
             // Warning: This is a performance killer. Real implementation uses warp-shuffles to reduce.
             
             for (int c = 0; c < NETWORK_WIDTH; ++c)
             {
                 NrcFloat w_grad = grad * layer_activations[layer][c];
                 int w_idx = weight_offset + r * NETWORK_WIDTH + c;
                 
                 // Using InterlockedAddFloat equivalent (requires CAS loop if not widely supported)
                 // or just write to a huge buffer [BatchSize][Weights] and reduce later.
                 // Let's assume we can tolerate race conditions for HOGWILD-style SGD,
                 // or we use a very small batch size per frame and do it serially?
                 // No, we need atomics.
                 
                 // Placeholder for Atomic Float Add:
                 // AtomicAddFloat(g_Gradients[w_idx], w_grad);
             }
             
             // Compute gradient for previous layer
             // dL_dX_prev_c += dL_dOutput_r * W_rc
             for (int c = 0; c < NETWORK_WIDTH; ++c)
             {
                 dL_dX_prev[c] += grad * g_Weights[weight_offset + r * NETWORK_WIDTH + c];
             }
         }
         
         // Move to next
         for(int k=0; k<NETWORK_WIDTH; ++k) dL_dX[k] = dL_dX_prev[k];
    }
}


/*

//run this after ray tracing...

// Requires -enable-16bit-types and shader model 6.2+
struct InitialGradientConstants {
    uint TotalPixels;
    float Epsilon;
};

ConstantBuffer<InitialGradientConstants> cb : register(b0);

// StructuredBuffers for 16-bit types
StructuredBuffer<float16_t2> global_outputs   : register(t0);
StructuredBuffer<float16_t2> global_targets   : register(t1);
RWStructuredBuffer<float16_t2> global_grads_out : register(u0);

[numthreads(256, 1, 1)]
void ComputeInitialGradientKernel(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;

    if (idx >= cb.TotalPixels) return;

    // Each 'pixel' has 3 colors (RGB) stored in two float16_t2 elements
    // Note: The 4th channel is typically padding or bias
    for (uint i = 0; i < 2; i++) {
        uint bufferIdx = idx * 2 + i;
        
        float16_t2 pred   = global_outputs[bufferIdx];
        float16_t2 target = global_targets[bufferIdx];

        // Step 1: Calculate (Prediction - Target)
        float16_t2 diff = pred - target;

        // Step 2: Calculate Derivative of Relative L2 Loss
        // Formula: 2 * (pred - target) / (pred^2 + epsilon)
        float16_t2 pred_sq    = pred * pred;
        float16_t2 epsilon    = (float16_t)cb.Epsilon;
        float16_t2 denominator = pred_sq + epsilon;
        
        // Final Initial Gradient
        global_grads_out[bufferIdx] = ((float16_t)2.0 * diff) / denominator;
    }
}

*/