/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

// #include "../../gpu_shared.h" // Removed to avoid include path issues. 
// Standard HLSL types used.

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

ConstantBuffer<NRCConstants> g_NRCConstants : register(b0);

// Weights buffer: [Layer 0 (64x64 + 64 bias) | Layer 1 | ... ]
// For simplicity, let's assume fully connected layers without explicit bias in the matrix for now (or bias concatenated)
// The reference kernel uses 64 inputs -> 64 outputs.
RWStructuredBuffer<NrcFloat> g_Weights : register(u0);

// Inference Inputs: [Pos (3), Dir (3), Supplemental (Normal, Roughness?)] -> Encoded to 64
// We will assume the input buffer already contains encoded features for this pass, 
// or we do on-the-fly encoding. Ideally on-the-fly.
// Let's store raw queries and encode in kernel.
struct InferenceQuery
{
    float3 pos;
    float3 dir;
    float3 normal;
    float roughness;
    float3 albedo;
    uint2 pixel_coord;
    float3 throughput; // Added
};

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


// --- Helper Functions ---
// OneBlob encoding or similar would go here. For now, a placeholder identity/frequency encoding.
// Mapping 3+3+3+1 = 10 input dims to 64 network dims.
// Simple frequency encoding: sin/cos of (P * freq), etc.
void EncodeInput(InferenceQuery query, out NrcFloat features[NETWORK_WIDTH])
{
    // Zero out
    for(int i=0; i<NETWORK_WIDTH; ++i) features[i] = 0.0f;

    // Simple frequency encoding for Pos (x,y,z)
    // 3 coords * 4 freqs * 2 (sin/cos) = 24
    // Dir (x,y,z) * 2 freqs * 2 = 12
    // Normal * 2 freqs * 2 = 12
    // Roughness * 4 = 4
    // Total used: 52. Pad rest.
    
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
    EncodeInput(q, features);
}

// Forward Pass
// Uses Shared Memory for weights to reuse across thread block.
// Groupshared: We need to load 64x64 float weights. 4KB. Fits easily.
groupshared NrcFloat s_LayerWeights[NETWORK_WIDTH * NETWORK_WIDTH];

[numthreads(GROUP_SIZE, 1, 1)]
void NRCInference(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint count = g_Counters[0];
    if (dtid.x >= count) return; // Dynamic count check

    InferenceQuery query = g_InferenceQueries[dtid.x];
    
    // 1. Encode
    NrcFloat activations[NETWORK_WIDTH];
    EncodeInput(query, activations);
    
    // 2. Run Layers
    for (int layer = 0; layer < HIDDEN_LAYERS + 2; ++layer) // +2 for In/Out?
    {
        // For simplicity, let's do 5 layers of 64x64.
        // Load Weights into Shared Memory cooperatively
        // Total weights: 64*64 = 4096.
        // Threads: 128. Each thread loads 32 floats.
        uint weight_offset = layer * NETWORK_WIDTH * NETWORK_WIDTH;
        
        for (int i = 0; i < 32; ++i)
        {
            int idx = i * GROUP_SIZE + gtid.x;
            if (idx < NETWORK_WIDTH * NETWORK_WIDTH)
            {
                s_LayerWeights[idx] = g_Weights[weight_offset + idx];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        
        // Matrix Multiply: Output = Activation * Weight
        // We calculate new activations for this thread
        NrcFloat next_activations[NETWORK_WIDTH];
        
        for (int out_row = 0; out_row < NETWORK_WIDTH; ++out_row)
        {
            NrcFloat sum = 0.0f;
            for (int in_col = 0; in_col < NETWORK_WIDTH; ++in_col)
            {
                // W is stored Row-Major? Let's assume W[row][col]
                // Output[row] = Sum(Input[col] * W[row][col])
                sum += activations[in_col] * s_LayerWeights[out_row * NETWORK_WIDTH + in_col];
            }
            
            // ReLU (except last layer?)
            // For now, ReLU all hidden.
            if (layer < HIDDEN_LAYERS)
                 next_activations[out_row] = max(0.0f, sum);
            else
                 next_activations[out_row] = sum; // Sigmoid or Linear for last?
        }
        
        // Update registers
        activations = next_activations;
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    // 3. Output
    // First 3 floats are RGB.
    // Factorize with Albedo (from query).
    float3 radiance = float3(activations[0], activations[1], activations[2]);
    
    // Apply albedo factorization (demodulation)
    // In paper: Network predicts "Radiance / Albedo" (plus some factor), so we multiply back.
    // Ensure positive
    radiance = max(0.0f, radiance);
    
    // Write Output
    // Simple Exponential accumulation check
    float3 final_color = radiance * (query.albedo + 0.001f); // Multiply by albedo
    
    // Add to Accumulation
    // Note: This is racy if multiple queries hit same pixel? 
    // Usually 1 query per pixel if bounce=2.
    // If we have jitter, we need to access the AccBuffer carefully.
    // But let's assume standard accumulation logic happens in RayGen for primary, here for indirect.
    // "radiance" here is L_indirect.
    // L_total = L_primary + Throughput * L_indirect
    
    float4 acc = g_AccumulationBuffer[query.pixel_coord];
    float3 current_radiance = acc.xyz;
    uint sample_count = asuint(acc.w);
    
    // Weighted update? The RayGen already divided L_primary by N.
    // We should divide this by N too.
    // BUT RayGen did: rad = rad_prev + (new - rad_prev)/N.
    // It's easier if RayGen writes (L_primary) and we ADD (Throughput * L_nrc / N).
    // Or we just strictly add to the current frame's radiance before averaging?
    // Let's assume RayGen wrote `Prev + (Primary - Prev)/N`.
    // We want `Prev + ((Primary + T*NRC) - Prev)/N`.
    // Difference is `(T*NRC)/N`.
    
    if (sample_count > 0)
        g_AccumulationBuffer[query.pixel_coord] = float4(current_radiance + (final_color * query.throughput / (float)sample_count), asfloat(sample_count));
    
    g_OutputTexture[query.pixel_coord] = float4(final_color, 1.0f);
}

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
