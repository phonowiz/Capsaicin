// inference kernel


#define LAYER_WIDTH_HALVES 32   // 64 neurons
#define THREAD_NEURONS_HALVES 8   // 16 neurons per thread (8 half2)
#define BLOCK_PIXELS 128        // 128 pixels per thread block

__global__ void FusedNRCKernel(half2* global_inputs, half2* global_outputs, half2* global_weights) {
    // 1. Identification
    const int pixel_in_block = threadIdx.x; // 0-127
    const int warp_id = threadIdx.x / 32;   // 0-3
    const int lane_id = threadIdx.x % 32;   // 0-31
    const int global_pixel_idx = blockIdx.x * BLOCK_PIXELS + pixel_in_block;

    // Shared memory for intermediate activations (64 neurons x 128 pixels)
    // This acts as the "hand-off" between layers
    __shared__ half2 smem_activations[LAYER_WIDTH_HALVES][BLOCK_PIXELS];
    __shared__ half2 smem_weights[LAYER_WIDTH_HALVES * 64]; // Current layer's weights

    // Load initial 64-dim input into registers, then move to Shared Memory
    #pragma unroll
    for (int i = 0; i < LAYER_WIDTH_HALVES; i++) {
        half2 val = global_inputs[global_pixel_idx * LAYER_WIDTH_HALVES + i];
        smem_activations[i][pixel_in_block] = val;
    }

    // --- 7 Layers of MLP ---
    for (int layer = 0; layer < 7; layer++) {
        // A. Cooperative weight load (128 threads load the 64x64 matrix)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int w_idx = i * BLOCK_PIXELS + pixel_in_block;
            smem_weights[w_idx] = global_weights[layer * 2048 + w_idx];
        }
        __syncthreads(); 

        // B. Computing the 16-neuron block-row assigned to this Warp
        // Notice: This loop only runs 8 times (8 half2 = 16 neurons)
        half2 my_results[THREAD_NEURONS_HALVES]; 
        
        #pragma unroll
        for (int out_pair = 0; out_pair < THREAD_NEURONS_HALVES; out_pair++) {
            half2 sum2 = __float2half2_rn(0.0f);
            
            // This warp-id logic ensures Warp 1 handles neurons 16-31, etc.
            int row_idx = (warp_id * THREAD_NEURONS_HALVES) + out_pair;

            #pragma unroll
            for (int in_pair = 0; in_pair < LAYER_WIDTH_HALVES; in_pair++) {
                half2 w = smem_weights[row_idx * LAYER_WIDTH_HALVES + in_pair];
                half2 a = smem_activations[in_pair][pixel_in_block];
                sum2 = __hfma2(a, w, sum2);
            }
            // ReLU
            my_results[out_pair] = __hmax2(sum2, __float2half2_rn(0.0f));
        }

        // C. Hand-off: Write 16 results back to Smem for the next layer's input
        __syncthreads(); // Wait for all warps to finish reading current layer
        #pragma unroll
        for (int out_pair = 0; out_pair < THREAD_NEURONS_HALVES; out_pair++) {
            int row_idx = (warp_id * THREAD_NEURONS_HALVES) + out_pair;
            smem_activations[row_idx][pixel_in_block] = my_results[out_pair];
        }
        __syncthreads(); // Wait for all warps to finish writing before next layer begins
    }

    // 3. Final Write-out (Reducing 64 neurons to 3 RGB values)
    // The network's 64 hidden neurons contain the final result.
    // Typically, the first 3 neurons are used for R, G, and B.
    // 3. Final Write-out with Reflectance Factorization
    if (warp_id == 0) {
        // 1. Fetch the raw neural predictions (Demodulated Radiance)
        half2 rg_raw = smem_activations[0][pixel_in_block]; // Neurons 0 & 1
        half  b_raw  = smem_activations[1][pixel_in_block].x; // Neuron 2 (Low half)

        // 2. Combine albedos (This is the "Factor")
        // Note: These should be the same values passed into the input encoding
        half3 albedo_sum = diffuse_albedo + specular_reflectance;

        // 3. Multiply (Factorization)
        half3 final_color;
        final_color.x = rg_raw.x * albedo_sum.x; // Red
        final_color.y = rg_raw.y * albedo_sum.y; // Green
        final_color.z = b_raw   * albedo_sum.z; // Blue

        // 4. Write to global memory
        // Pack back to half2 for the output buffer if needed
        global_outputs[global_pixel_idx * 2]     = __halves2half2(final_color.x, final_color.y);
        global_outputs[global_pixel_idx * 2 + 1] = __halves2half2(final_color.z, __float2half(0.0f));
    }
}



//backprop pass


// smem_activations: [64 neurons][128 pixels]
// smem_dL_da: [64 neurons][128 pixels]
__device__ void compute_weight_gradient(
    half2 smem_activations[32][128], 
    half2 smem_dL_da[32][128], 
    half2* global_delta_w_ptr
) {
    const int warp_id = threadIdx.x / 32; // 0-3
    const int lane_id = threadIdx.x % 32; // 0-31

    // Each thread stores 8 partial sums (16 neurons) in registers.
    // This thread will contribute to one specific 'column' (lane_id) 
    // for its warp's assigned 'rows' (warp_id).
    half2 local_delta_w[8]; 

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        local_delta_w[i] = __float2half2_rn(0.0f);
    }

    // Loop over the 128 pixels (the batch dimension)
    for (int pix = 0; pix < 128; pix++) {
        // The gradient for the 'output' neuron this thread is responsible for
        // In this tiling, each thread in the warp handles one column of the weight matrix
        half2 grad = smem_dL_da[lane_id][pix];

        #pragma unroll
        //go down the column of neurons for this pixel
        for (int i = 0; i < 8; i++) {

            // Reading A[row_idx][pix] 
            // By using 'pix' as the shared index, we are effectively 
            // treating the row of A as a column of A^T.

            // The activation of the 'input' neuron
            int row_idx = (warp_id * 8) + i;
            half2 act = smem_activations[row_idx][pix];

            // Outer product: DeltaW = Act * Grad
            // This determines how much this specific weight influenced the error
            local_delta_w[i] = __hfma2(act, grad, local_delta_w[i]);
        }
    }

    // Write the accumulated gradients back to Global Memory
    // Note: Since multiple thread blocks process different sets of 128 pixels,
    // we use atomicAdd to sum them all into the final gradient buffer.
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = (warp_id * 8) + i;
        int col = lane_id;
        int global_idx = row * 32 + col;
        
        // atomicAdd for half2 requires compute capability 7.0+
        atomicAdd(&global_delta_w_ptr[global_idx], local_delta_w[i]);
    }
}


// // 64x64 matrix = 4096 elements. Using half2, that's 2048 elements.
// // The thread block has 128 threads.
// // Each thread must load: 2048 / 128 = 16 half2 elements.
// __device__ void load_weights_transposed(half2* smem_weights_T, half2* global_weights_ptr) {
//     const int tid = threadIdx.x; // 0-127

//     #pragma unroll
//     for (int i = 0; i < 16; i++) {
//         int linear_idx = (i * 128) + tid;
        
//         // Logical coordinates in original matrix
//         int row = linear_idx / 32; 
//         int col = linear_idx % 32;

//         // Store into Shared Memory in Transposed position [col][row]
//         // This makes 'get_weight_transpose' a simple array lookup later
//         smem_weights_T[col * 64 + row] = global_weights_ptr[linear_idx];
//     }


//     __syncthreads();
// }

#define LAYER_WIDTH_HALVES 32   // 64 neurons
#define THREAD_NEURONS_HALVES 8   // 16 neurons per thread
#define BLOCK_PIXELS 128        // 128 pixels per thread block

__global__ void FusedNRC_BackwardKernel(
    half2* global_grads_out,   // Incoming gradients from next layer
    half2* global_activations, // Stored from forward pass
    half2* global_weights, 
    half2* global_delta_w      // Output: accumulated weight gradients
) {
    const int pixel_in_block = threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_pixel_idx = blockIdx.x * BLOCK_PIXELS + pixel_in_block;

    // Shared memory buffers
    __shared__ half2 smem_dL_da[LAYER_WIDTH_HALVES][BLOCK_PIXELS];
    __shared__ half2 smem_activations[LAYER_WIDTH_HALVES][BLOCK_PIXELS];
    __shared__ half2 smem_weights_T[LAYER_WIDTH_HALVES * 64]; 

    // 1. Initial Load: Gradients from the output layer
    #pragma unroll
    for (int i = 0; i < LAYER_WIDTH_HALVES; i++) {
        smem_dL_da[i][pixel_in_block] = global_grads_out[global_pixel_idx * LAYER_WIDTH_HALVES + i];
    }
    __syncthreads();

    // 2. Backprop Loop (7 layers)
    for (int layer = 6; layer >= 0; layer--) {
        
        // --- A. LOAD PHASE ---
        // Load activations from forward pass
        #pragma unroll
        for (int i = 0; i < LAYER_WIDTH_HALVES; i++) {
            smem_activations[i][pixel_in_block] = global_activations[(layer * TOTAL_PIXELS + global_pixel_idx) * LAYER_WIDTH_HALVES + i];
        }

        // Load weights and TRANSPOSE them into shared memory
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int tid = threadIdx.x;
            int linear_idx = (i * 128) + tid;
            int row = linear_idx / 32; 
            int col = linear_idx % 32;
            // Store W[row][col] into smem[col][row]
            smem_weights_T[col * 64 + row] = global_weights[layer * 2048 + linear_idx];
        }
        __syncthreads();

        // --- B. COMPUTE WEIGHT GRADIENTS (dl_dW = X^T * dl_da) ---
        half2 local_delta_w[THREAD_NEURONS_HALVES];
        #pragma unroll
        for (int i = 0; i < THREAD_NEURONS_HALVES; i++) local_delta_w[i] = __float2half2_rn(0.0f);

        for (int pix = 0; pix < BLOCK_PIXELS; pix++) {
            half2 grad = smem_dL_da[lane_id][pix];
            #pragma unroll
            for (int i = 0; i < THREAD_NEURONS_HALVES; i++) {
                int row_idx = (warp_id * THREAD_NEURONS_HALVES) + i;
                half2 act = smem_activations[row_idx][pix];
                local_delta_w[i] = __hfma2(act, grad, local_delta_w[i]);
            }
        }

        // Write weight gradients to global memory using atomics
        #pragma unroll
        for (int i = 0; i < THREAD_NEURONS_HALVES; i++) {
            int row = (warp_id * THREAD_NEURONS_HALVES) + i;
            int global_idx = layer * 2048 + (row * 32 + lane_id);
            atomicAdd(&global_delta_w[global_idx], local_delta_w[i]); //global_delta_w = dL/dW (how much each weight should change)
        }

        // --- C. COMPUTE INPUT GRADIENTS (dL/dX = G * W^T) with ReLU Derivative ---
        half2 next_grads[THREAD_NEURONS_HALVES];
        #pragma unroll
        // This loop calculates the weighted contribution part of dY/dX
        //Y = ReLU(W * X + b)
        for (int in_p = 0; in_p < THREAD_NEURONS_HALVES; in_p++) {
            half2 g_sum = __float2half2_rn(0.0f);
            #pragma unroll
            for (int out_p = 0; out_p < LAYER_WIDTH_HALVES; out_p++) {
                half2 w_t = smem_weights_T[in_p * 64 + out_p];
                half2 g_up = smem_dL_da[out_p][pixel_in_block];  //this is delta of the "next" layer which we already processed
                g_sum = __hfma2(g_up, w_t, g_sum); //this is the dot product of the next layer's delta and the weights of this layer
            }
            
            /*
                The second part of $dY/dX$ is the derivative of the activation function. For ReLU, the derivative is 1$1$ if the input was positive and 2$0$ otherwise.3 This is the "masking" logic found at the end of Step C:
            */
            int act_idx = (warp_id * THREAD_NEURONS_HALVES) + in_p;
            half2 fwd_act = smem_activations[act_idx][pixel_in_block];
            half2 mask = __hgt2(fwd_act, __float2half2_rn(0.0f));
            next_grads[in_p] = __hmul2(g_sum, mask); //dL/dX (Gradients for the inputs of this layer)
        }

        // --- D. HAND-OFF ---
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < THREAD_NEURONS_HALVES; i++) {
            int row_idx = (warp_id * THREAD_NEURONS_HALVES) + i;
            smem_dL_da[row_idx][pixel_in_block] = next_grads[i];
        }
        __syncthreads();
    }
}


__global__ void ComputeInitialGradientKernel(
    half2* global_outputs,    // Raw predictions from the Forward Pass
    half2* global_targets,    // The path-traced "Ground Truth"
    half2* global_grads_out,  // Buffer to fill for the Backward Pass
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Each 'pixel' has 3 colors (RGB) stored in two half2 elements
    for (int i = 0; i < 2; i++) {
        half2 pred = global_outputs[idx * 2 + i];
        half2 target = global_targets[idx * 2 + i];

        // Step 1: Calculate (Prediction - Target)
        half2 diff = __hsub2(pred, target);

        // Step 1: Scale by 2 / (Pred^2 + epsilon) 
        // This is the derivative of the Relative L2 Loss
        half2 pred_sq = __hmul2(pred, pred);
        half2 epsilon = __float2half2_rn(0.01f);
        half2 denominator = __hadd2(pred_sq, epsilon);
        
        // Final Initial Gradient G
        global_grads_out[idx * 2 + i] = __h2div(__hmul2(__float2half2_rn(2.0f), diff), denominator);
    }
}


__global__ void ApplyWeightCorrection(
    half2* weights,          // The actual network weights
    half2* accumulated_grads,// The buffer filled by compute_weight_gradient -> same as global_delta_w
    half2* first_moments,    // For Adam optimizer (m)
    half2* second_moments,   // For Adam optimizer (v)
    float learning_rate,
    int total_weights_halves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_weights_halves) return;

    // 1. Fetch the total gradient for this weight
    half2 grad = accumulated_grads[idx];

    // 2. Simple SGD update logic (or insert Adam logic here)
    // weights[idx] = weights[idx] - (learning_rate * grad)
    half2 current_w = weights[idx];
    half2 correction = __hmul2(__float2half2_rn(learning_rate), grad);
    weights[idx] = __hsub2(current_w, correction);

    // 3. CRITICAL: Clear the gradient buffer for the next frame
    accumulated_grads[idx] = __float2half2_rn(0.0f);
}