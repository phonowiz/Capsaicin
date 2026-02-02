/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#include "NeuralRadianceCache.h"

// Include for RayCamera definition and helper
#include "../../ray_tracing/path_tracing_shared.h"

#include "components/light_builder/light_builder.h"
#include "components/light_sampler/light_sampler_switcher.h"
#include "components/random_number_generator/random_number_generator.h"
#include "components/stratified_sampler/stratified_sampler.h"

#include <random>

namespace Capsaicin
{
NeuralRadianceCache::NeuralRadianceCache()
    : RenderTechnique("Neural Radiance Cache")
{}

NeuralRadianceCache::~NeuralRadianceCache()
{
    NeuralRadianceCache::terminate();
}

RenderOptionList NeuralRadianceCache::getRenderOptions() noexcept
{
    RenderOptionList newOptions;
    newOptions.emplace(RENDER_OPTION_MAKE(nrc_train_active, options));
    newOptions.emplace(RENDER_OPTION_MAKE(nrc_inference_active, options));
    newOptions.emplace(RENDER_OPTION_MAKE(nrc_learning_rate, options));
    newOptions.emplace(RENDER_OPTION_MAKE(nrc_batch_size, options));
    return newOptions;
}

NeuralRadianceCache::RenderOptions NeuralRadianceCache::convertOptions(RenderOptionList const &opt) noexcept
{
    RenderOptions newOptions;
    RENDER_OPTION_GET(nrc_train_active, newOptions, opt)
    RENDER_OPTION_GET(nrc_inference_active, newOptions, opt)
    RENDER_OPTION_GET(nrc_learning_rate, newOptions, opt)
    RENDER_OPTION_GET(nrc_batch_size, newOptions, opt)
    return newOptions;
}

ComponentList NeuralRadianceCache::getComponents() const noexcept
{
    ComponentList components;
    components.emplace_back(COMPONENT_MAKE(LightSamplerSwitcher));
    components.emplace_back(COMPONENT_MAKE(StratifiedSampler));
    components.emplace_back(COMPONENT_MAKE(RandomNumberGenerator));
    return components;
}

SharedTextureList NeuralRadianceCache::getSharedTextures() const noexcept
{
    SharedTextureList textures;
    // textures.push_back({"NRC_Output", SharedTexture::Access::Write}); // Internal only, don't declare as required shared
    textures.push_back({"Color", SharedTexture::Access::Write}); // Output of PT
    return textures;
}

bool NeuralRadianceCache::init(CapsaicinInternal const &capsaicin) noexcept
{
    // Initialize Buffers (7 layers of 64x64 weights)
    uint32_t num_layers = 7;
    uint32_t weight_count = num_layers * 64 * 64; 
    
    // Simple float-to-half conversion helper for initialization
    auto floatToHalf = [](float f) -> uint16_t {
        uint32_t i = *((uint32_t*)&f);
        uint32_t s = (i >> 16) & 0x00008000;
        uint32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
        uint32_t m = i & 0x007fffff;
        if (e <= 0) return (uint16_t)s;
        if (e >= 31) return (uint16_t)(s | 0x7c00);
        return (uint16_t)(s | (e << 10) | (m >> 13));
    };

    // Init Weights on CPU with Xavier/He initialization (as half-precision)
    std::vector<uint16_t> initial_weights(weight_count);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f); 
    for(auto& w : initial_weights) w = floatToHalf(dist(rng));

    weights_buffer_ = gfxCreateBuffer(gfx_, weight_count * sizeof(uint16_t), initial_weights.data(), kGfxCpuAccess_None);
    weights_buffer_.setName("NRC_Weights");
    
    gradients_buffer_ = gfxCreateBuffer(gfx_, weight_count * sizeof(uint16_t), nullptr, kGfxCpuAccess_None);
    gradients_buffer_.setName("NRC_Gradients");
    
    momentum1_buffer_ = gfxCreateBuffer(gfx_, weight_count * sizeof(uint16_t), nullptr, kGfxCpuAccess_None);
    momentum2_buffer_ = gfxCreateBuffer(gfx_, weight_count * sizeof(uint16_t), nullptr, kGfxCpuAccess_None);

    // Queries/Samples
    uint32_t max_queries = 1920 * 1080;

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

    uint32_t struct_size = sizeof(InferenceQuery);

    inference_queries_ = gfxCreateBuffer(gfx_, max_queries * struct_size, nullptr, kGfxCpuAccess_None);
    inference_queries_.setName("NRC_InferenceQueries");
    
    // struct_size is already sizeof(InferenceQuery)
    training_samples_ = gfxCreateBuffer(gfx_, max_queries * struct_size, nullptr, kGfxCpuAccess_None);
    training_samples_.setName("NRC_TrainingSamples");

    activations_buffer_ = gfxCreateBuffer(gfx_, 7 * max_queries * 64 * sizeof(uint16_t), nullptr, kGfxCpuAccess_None);
    activations_buffer_.setName("NRC_Activations");

    incoming_gradients_ = gfxCreateBuffer(gfx_, max_queries * 64 * sizeof(uint16_t), nullptr, kGfxCpuAccess_None);
    incoming_gradients_.setName("NRC_IncomingGradients");
    
    counters_buffer_ = gfxCreateBuffer(gfx_, 2 * sizeof(uint32_t), nullptr, kGfxCpuAccess_None); 
    counters_buffer_.setName("NRC_Counters");

    constants_buffer_ = gfxCreateBuffer<NRCConstants>(gfx_, 1, nullptr, kGfxCpuAccess_Write);
    constants_buffer_.setName("NRC_Constants");

    output_texture_ = capsaicin.createRenderTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, "NRC_Output");
    
    ray_camera_buffer_ = gfxCreateBuffer<RayCamera>(gfx_, 1, nullptr, kGfxCpuAccess_Write);
    ray_camera_buffer_.setName("NRC_RayCamera");

    nrc_inference_program_ = capsaicin.createProgram("render_techniques/neural_radiance_cache/nrc_inference");
    nrc_train_program_ = capsaicin.createProgram("render_techniques/neural_radiance_cache/nrc_train");
    nrc_loss_program_ = capsaicin.createProgram("render_techniques/neural_radiance_cache/nrc_loss");
    nrc_adam_program_ = capsaicin.createProgram("render_techniques/neural_radiance_cache/nrc_adam_optimizer");

    adam_constants_buffer_ = gfxCreateBuffer<AdamConstants>(gfx_, 1, nullptr, kGfxCpuAccess_Write);
    adam_constants_buffer_.setName("NRC_AdamConstants");

    return initKernels(capsaicin);
}

bool NeuralRadianceCache::initKernels(CapsaicinInternal const &capsaicin) noexcept
{
    // Init Compute
    inference_kernel_ = gfxCreateComputeKernel(gfx_, nrc_inference_program_, "NRCInference");
    train_kernel_ = gfxCreateComputeKernel(gfx_, nrc_train_program_, "NRCTrain");
    nrc_loss_kernel_ = gfxCreateComputeKernel(gfx_, nrc_loss_program_, "NRCLoss");
    adam_kernel_ = gfxCreateComputeKernel(gfx_, nrc_adam_program_, "main");
    
    // Init RT
    // Need setupRTKernel equivalent
    std::vector<char const *> exports;
    std::vector<char const *> subobjects;
    std::vector<std::string>  defines_str;
    std::vector<std::string>  exports_str;
    std::vector<std::string>  subobjects_str;
    std::vector<GfxLocalRootSignatureAssociation> local_root_signature_associations;
    
    // Defines
    auto const lightSampler = capsaicin.getComponent<LightSamplerSwitcher>();
    defines_str = lightSampler->getShaderDefines(capsaicin);
    
    setupRTKernel(capsaicin, local_root_signature_associations, defines_str, exports_str, subobjects_str);
    
    std::vector<char const *> defines;
    for(auto& s : defines_str) defines.push_back(s.c_str());
    for(auto& s : exports_str) exports.push_back(s.c_str());
    for(auto& s : subobjects_str) subobjects.push_back(s.c_str());

    rt_program_ = capsaicin.createProgram("render_techniques/neural_radiance_cache/nrc_path_tracer"); // Maps to .rt file
    
    rt_kernel_ = gfxCreateRaytracingKernel(gfx_, rt_program_,
        local_root_signature_associations.data(), (uint32_t)local_root_signature_associations.size(),
        exports.data(), (uint32_t)exports.size(),
        subobjects.data(), (uint32_t)subobjects.size(),
        defines.data(), (uint32_t)defines.size());

    uint32_t entry_count[kGfxShaderGroupType_Count] {
        capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Raygen),
        capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Miss), 
        gfxSceneGetInstanceCount(capsaicin.getScene()) * capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Hit), 
        capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Callable)};
        
    GfxKernel sbt_kernels[] {rt_kernel_};
    rt_sbt_ = gfxCreateSbt(gfx_, sbt_kernels, ARRAYSIZE(sbt_kernels), entry_count);
    
    return !!nrc_inference_program_ && !!nrc_train_program_ && !!nrc_loss_program_ && !!nrc_adam_program_ && !!rt_program_;
}

void NeuralRadianceCache::render(CapsaicinInternal &capsaicin) noexcept
{
    RenderOptions const newOptions = convertOptions(capsaicin.getOptions());
    options = newOptions;
    
    // Update Camera
    auto const camera = capsaicin.getCamera();
    auto const renderDimensions = capsaicin.getRenderDimensions();
    // Using caclulateRayCamera (typo) from path_tracing_shared.h
    RayCamera cameraData = caclulateRayCamera(
        {camera.eye, camera.center, camera.up, camera.aspect, camera.fovY, camera.nearZ, camera.farZ},
        renderDimensions);

    // 0. Clear Counters
    // Use gfxCommandClearBuffer to clear to 0
    gfxCommandClearBuffer(gfx_, counters_buffer_, 0); 
    
    // 0b. Resize check (simplified: assume fixed max)

    // 1. Run Path Tracer
    // Bind Params
    gfxProgramSetParameter(gfx_, rt_program_, "g_BufferDimensions", renderDimensions);
    gfxProgramSetParameter(gfx_, rt_program_, "g_FrameIndex", capsaicin.getFrameIndex());
    gfxProgramSetParameter(gfx_, rt_program_, "g_RayCamera", cameraData);
    gfxProgramSetParameter(gfx_, rt_program_, "g_BounceCount", 5); // Hardcoded for now
    gfxProgramSetParameter(gfx_, rt_program_, "g_BounceRRCount", 2);
    gfxProgramSetParameter(gfx_, rt_program_, "g_Accumulate", 1);
    
    auto const stratified_sampler = capsaicin.getComponent<StratifiedSampler>();
    auto const rng = capsaicin.getComponent<RandomNumberGenerator>();
    auto const lightSampler = capsaicin.getComponent<LightSamplerSwitcher>();

    stratified_sampler->addProgramParameters(capsaicin, rt_program_);
    rng->addProgramParameters(capsaicin, rt_program_);
    lightSampler->addProgramParameters(capsaicin, rt_program_);

    gfxProgramSetParameter(gfx_, rt_program_, "g_InstanceBuffer", capsaicin.getInstanceBuffer());
    gfxProgramSetParameter(gfx_, rt_program_, "g_TransformBuffer", capsaicin.getTransformBuffer());
    gfxProgramSetParameter(gfx_, rt_program_, "g_IndexBuffer", capsaicin.getIndexBuffer());
    gfxProgramSetParameter(gfx_, rt_program_, "g_VertexBuffer", capsaicin.getVertexBuffer());
    gfxProgramSetParameter(gfx_, rt_program_, "g_VertexDataIndex", capsaicin.getVertexDataIndex());
    gfxProgramSetParameter(gfx_, rt_program_, "g_MaterialBuffer", capsaicin.getMaterialBuffer());
    gfxProgramSetParameter(gfx_, rt_program_, "g_Scene", capsaicin.getAccelerationStructure());
    gfxProgramSetParameter(gfx_, rt_program_, "g_EnvironmentBuffer", capsaicin.getEnvironmentBuffer());
    auto const &textures = capsaicin.getTextures();
    gfxProgramSetParameter(gfx_, rt_program_, "g_TextureMaps", textures.data(), static_cast<uint32_t>(textures.size()));
    gfxProgramSetParameter(gfx_, rt_program_, "g_TextureSampler", capsaicin.getLinearWrapSampler());
    
    // Output
    if (output_texture_.getWidth() != renderDimensions.x || output_texture_.getHeight() != renderDimensions.y)
    {
        gfxDestroyTexture(gfx_, output_texture_);
        output_texture_ = capsaicin.createRenderTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, "NRC_Output");
    }
    
    gfxProgramSetParameter(gfx_, rt_program_, "g_OutputBuffer", capsaicin.getSharedTexture("Color"));
    
    // BIND NRC BUFFERS TO RT
    gfxProgramSetParameter(gfx_, rt_program_, "g_InferenceQueries_RT", inference_queries_);
    gfxProgramSetParameter(gfx_, rt_program_, "g_TrainingSamples_RT", training_samples_);
    gfxProgramSetParameter(gfx_, rt_program_, "g_Counters", counters_buffer_);

    setupSbt(capsaicin);
    gfxCommandBindKernel(gfx_, rt_kernel_);
    gfxCommandDispatchRays(gfx_, rt_sbt_, renderDimensions.x, renderDimensions.y, 1);

    // Barrier
    // gfxCommandMemoryBarrier? Not exposed. Assume implicit.

    // 2. Dispatch Inference
    if (options.nrc_inference_active)
    {
       // Update Constants
       NRCConstants *constants = gfxBufferGetData<NRCConstants>(gfx_, constants_buffer_);
       constants->num_training_samples = 1920 * 1080; // Full screen batch size
       constants->num_inference_queries = renderDimensions.x * renderDimensions.y;
       constants->learning_rate = options.nrc_learning_rate;
       constants->batch_size = options.nrc_batch_size;
       constants->activations_stride = 1920 * 1080; // max_queries
       constants->activations_offset = 0;
       constants->is_training_pass = 0;

       // Bind Inference Parameters
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_NRCConstants", constants_buffer_);   // b0
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_Weights", weights_buffer_);           // t1
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_InferenceQueries", inference_queries_); // t0
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_Counters", counters_buffer_);         // t3
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_OutputTexture", capsaicin.getSharedTexture("Color"));     // u1
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_Activations", activations_buffer_);     // u2
    
       uint32_t num_queries = renderDimensions.x * renderDimensions.y;
       uint32_t num_groups = (num_queries + 127) / 128;
       gfxCommandBindKernel(gfx_, inference_kernel_);
       gfxCommandDispatch(gfx_, num_groups, 1, 1);
    }

    // 2.5 Dispatch Forward Pass for Training Samples (to get activations)
    if (options.nrc_train_active)
    {
       NRCConstants *constants = gfxBufferGetData<NRCConstants>(gfx_, constants_buffer_);
       // Re-update constants for training pass
       constants->is_training_pass = 1;
       
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_NRCConstants", constants_buffer_);
       gfxProgramSetParameter(gfx_, nrc_inference_program_, "g_InferenceQueries", training_samples_); // REBIND to training samples
       
       // Assuming 'training_samples_' has same size/layout as inference queries for safety, or at least sufficient for max samples
       uint32_t num_training_samples = 20736;
       // Note: counters_buffer_[1] has actual count, but compute shader guards against out of bounds.
       uint32_t num_groups_train_infer = (num_training_samples + 127) / 128;
       
       gfxCommandBindKernel(gfx_, inference_kernel_);
       gfxCommandDispatch(gfx_, num_groups_train_infer, 1, 1);
    }

    // 2.5 Dispatch Loss
    if (options.nrc_train_active)
    {
        gfxProgramSetParameter(gfx_, nrc_loss_program_, "g_NRCConstants", constants_buffer_);
        gfxProgramSetParameter(gfx_, nrc_loss_program_, "g_TrainingSamples", training_samples_);
        gfxProgramSetParameter(gfx_, nrc_loss_program_, "g_Counters", counters_buffer_);
        gfxProgramSetParameter(gfx_, nrc_loss_program_, "g_Activations", activations_buffer_);
        gfxProgramSetParameter(gfx_, nrc_loss_program_, "g_IncomingGradients", incoming_gradients_);

        uint32_t batch_size = 20736;
        uint32_t num_groups_loss = (batch_size + 63) / 64;
        gfxCommandBindKernel(gfx_, nrc_loss_kernel_);
        gfxCommandDispatch(gfx_, num_groups_loss, 1, 1);
    }

    // 3. Dispatch Training
    if (options.nrc_train_active)
    {
        // Bind Training Parameters
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_NRCConstants", constants_buffer_);     // b0
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_Weights", weights_buffer_);             // t1
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_TrainingSamples", training_samples_);   // t2
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_WeightGradients", gradients_buffer_);    // u3
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_Momentum1", momentum1_buffer_);         // u4
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_Momentum2", momentum2_buffer_);         // u5
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_Counters", counters_buffer_);           // t3
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_IncomingGradients", incoming_gradients_); // u7
        gfxProgramSetParameter(gfx_, nrc_train_program_, "g_Activations", activations_buffer_);     // u2

        gfxCommandBindKernel(gfx_, train_kernel_);
        uint32_t batch_size       = 20736;
        uint32_t num_groups_train = (batch_size + 63) / 64; // Block size is 64 in nrc_train.comp
        gfxCommandDispatch(gfx_, num_groups_train, 1, 1);

        // 4. Dispatch Adam Optimizer
        step_count++;
        AdamConstants *adam_constants = gfxBufferGetData<AdamConstants>(gfx_, adam_constants_buffer_);
        adam_constants->learningRate = options.nrc_learning_rate;
        adam_constants->beta1        = 0.9f;
        adam_constants->beta2        = 0.99f;
        adam_constants->epsilon      = 1e-8f;
        adam_constants->t            = step_count;

        gfxProgramSetParameter(gfx_, nrc_adam_program_, "g_AdamConstants", adam_constants_buffer_);
        gfxProgramSetParameter(gfx_, nrc_adam_program_, "g_Weights", weights_buffer_);
        gfxProgramSetParameter(gfx_, nrc_adam_program_, "g_Gradients", gradients_buffer_);
        gfxProgramSetParameter(gfx_, nrc_adam_program_, "g_Momentum1", momentum1_buffer_);
        gfxProgramSetParameter(gfx_, nrc_adam_program_, "g_Momentum2", momentum2_buffer_);

        uint32_t num_weights = 7 * 64 * 64; // TOTAL_WEIGHT_ELEMENTS / 2 because we use float16_t2
        uint32_t weight_pairs = num_weights / 2;
        uint32_t num_groups_adam = (weight_pairs + 255) / 256;
        gfxCommandBindKernel(gfx_, adam_kernel_);
        gfxCommandDispatch(gfx_, num_groups_adam, 1, 1);
    }


}

void NeuralRadianceCache::setupSbt(CapsaicinInternal const &capsaicin) const noexcept
{
    auto const *kReferencePTRaygenShaderName       = "NRCPTRaygen";
    auto const *kReferencePTMissShaderName         = "NRCPTMiss";
    auto const *kReferencePTShadowMissShaderName   = "NRCPTShadowMiss";
    // auto const *kReferencePTHitGroupName           = "ReferencePTHitGroup"; // Reuse name if defined in subobjects
    // auto const *kReferencePTShadowHitGroupName     = "ReferencePTShadowHitGroup";

    gfxSbtSetShaderGroup(gfx_, rt_sbt_, kGfxShaderGroupType_Raygen, 0, kReferencePTRaygenShaderName);
    gfxSbtSetShaderGroup(gfx_, rt_sbt_, kGfxShaderGroupType_Miss, 0, kReferencePTMissShaderName);
    gfxSbtSetShaderGroup(gfx_, rt_sbt_, kGfxShaderGroupType_Miss, 1, kReferencePTShadowMissShaderName);

    for (uint32_t i = 0; i < capsaicin.getRaytracingPrimitiveCount(); i++)
    {
        gfxSbtSetShaderGroup(gfx_, rt_sbt_, kGfxShaderGroupType_Hit,
            i * capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Hit) + 0, "NRCHitGroup"); 
        gfxSbtSetShaderGroup(gfx_, rt_sbt_, kGfxShaderGroupType_Hit,
            i * capsaicin.getSbtStrideInEntries(kGfxShaderGroupType_Hit) + 1, "NRCShadowHitGroup");
    }
}

void NeuralRadianceCache::setupRTKernel([[maybe_unused]] CapsaicinInternal const &capsaicin,
    [[maybe_unused]] std::vector<GfxLocalRootSignatureAssociation> &local_root_signature_associations,
    [[maybe_unused]] std::vector<std::string> &defines, std::vector<std::string> &exports,
    std::vector<std::string> &subobjects) noexcept
{
    exports.emplace_back("NRCPTRaygen");
    exports.emplace_back("NRCPTMiss");
    exports.emplace_back("NRCPTShadowMiss");
    exports.emplace_back("NRCPTAnyHit");
    exports.emplace_back("NRCPTClosestHit");
    exports.emplace_back("NRCPTShadowAnyHit");

    subobjects.emplace_back("NRCShaderConfig");
    subobjects.emplace_back("NRCPipelineConfig");
    subobjects.emplace_back("NRCHitGroup");
    subobjects.emplace_back("NRCShadowHitGroup");
}

void NeuralRadianceCache::terminate() noexcept
{
    gfxDestroyBuffer(gfx_, weights_buffer_);
    gfxDestroyBuffer(gfx_, gradients_buffer_);
    gfxDestroyBuffer(gfx_, inference_queries_);
    gfxDestroyBuffer(gfx_, training_samples_);
    gfxDestroyBuffer(gfx_, counters_buffer_);
    gfxDestroyBuffer(gfx_, ray_camera_buffer_);
    gfxDestroyBuffer(gfx_, momentum1_buffer_);
    gfxDestroyBuffer(gfx_, momentum2_buffer_);
    gfxDestroyBuffer(gfx_, incoming_gradients_);

    gfxDestroyTexture(gfx_, output_texture_);
    
    gfxDestroyProgram(gfx_, nrc_inference_program_);
    gfxDestroyProgram(gfx_, nrc_train_program_);
    gfxDestroyProgram(gfx_, nrc_loss_program_);
    gfxDestroyProgram(gfx_, nrc_adam_program_);
    gfxDestroyKernel(gfx_, inference_kernel_);
    gfxDestroyKernel(gfx_, train_kernel_);
    gfxDestroyKernel(gfx_, nrc_loss_kernel_);
    gfxDestroyKernel(gfx_, adam_kernel_);
    gfxDestroyBuffer(gfx_, adam_constants_buffer_);
    
    gfxDestroyProgram(gfx_, rt_program_);
    gfxDestroyKernel(gfx_, rt_kernel_);
    gfxDestroySbt(gfx_, rt_sbt_);
}

void NeuralRadianceCache::renderGUI(CapsaicinInternal &capsaicin) const noexcept
{
    ImGui::Checkbox("NRC Inference", &capsaicin.getOption<bool>("nrc_inference_active"));
    ImGui::Checkbox("NRC Train", &capsaicin.getOption<bool>("nrc_train_active"));
    ImGui::DragFloat("Learning Rate", &capsaicin.getOption<float>("nrc_learning_rate"), 0.0001f, 0.0f, 1.0f, "%.5f");
}

} // namespace Capsaicin
