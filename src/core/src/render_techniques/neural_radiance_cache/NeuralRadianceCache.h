/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/
#pragma once

#include "render_techniques/render_technique.h"
#include "capsaicin_internal.h"

namespace Capsaicin
{
class NeuralRadianceCache : public RenderTechnique
{
public:
    NeuralRadianceCache();
    ~NeuralRadianceCache() override;

    NeuralRadianceCache(NeuralRadianceCache const &)            = delete;
    NeuralRadianceCache &operator=(NeuralRadianceCache const &) = delete;

    /* RenderTechnique Overrides */
    RenderOptionList getRenderOptions() noexcept override;
    struct RenderOptions
    {
        bool     nrc_train_active     = true;
        bool     nrc_inference_active = true;
        float    nrc_learning_rate    = 0.001f;
        uint32_t nrc_batch_size       = 1024;
    };

    struct NRCConstants
    {
        uint32_t num_training_samples;
        uint32_t num_inference_queries;
        float    learning_rate;
        uint32_t batch_size;
        uint32_t activations_stride;
        uint32_t activations_offset;
    };

    struct AdamConstants
    {
        float    learningRate;
        float    beta1;
        float    beta2;
        float    epsilon;
        uint32_t t; // Current training step/frame count
    };
    RenderOptions convertOptions(RenderOptionList const &options) noexcept;

    ComponentList getComponents() const noexcept override;
    SharedTextureList getSharedTextures() const noexcept override;

    bool init(CapsaicinInternal const &capsaicin) noexcept override;
    void render(CapsaicinInternal &capsaicin) noexcept override;
    void terminate() noexcept override;
    void renderGUI(CapsaicinInternal &capsaicin) const noexcept override;

private:
    GfxBuffer weights_buffer_;
    GfxBuffer gradients_buffer_;
    GfxBuffer momentum1_buffer_;
    GfxBuffer momentum2_buffer_;
    GfxBuffer inference_queries_;
    GfxBuffer training_samples_;
    GfxBuffer constants_buffer_;
    GfxBuffer activations_buffer_;
    GfxBuffer incoming_gradients_;
    GfxBuffer adam_constants_buffer_;

    GfxProgram nrc_inference_program_;
    GfxProgram nrc_train_program_;
    GfxProgram nrc_loss_program_;
    GfxProgram nrc_adam_program_;
    GfxKernel  inference_kernel_;
    GfxKernel  train_kernel_;
    GfxKernel  nrc_loss_kernel_;
    GfxKernel  adam_kernel_;
    GfxKernel  update_weights_kernel_; // Optional if integrated into train

    GfxTexture output_texture_; // Stores NRC inference result
    GfxBuffer  counters_buffer_; // [0] = Query Count, [1] = Sample Count

    // Compute Checks


    // RT Members
    GfxProgram rt_program_;
    GfxKernel  rt_kernel_;
    GfxSbt     rt_sbt_;
    GfxBuffer  ray_camera_buffer_;
    uint32_t   step_count = 0;

    RenderOptions options;

    bool initKernels(CapsaicinInternal const &capsaicin) noexcept;
    void setupSbt(CapsaicinInternal const &capsaicin) const noexcept;
    void setupRTKernel(CapsaicinInternal const &capsaicin,
        std::vector<GfxLocalRootSignatureAssociation> &local_root_signature_associations,
        std::vector<std::string> &defines, std::vector<std::string> &exports,
        std::vector<std::string> &subobjects) noexcept;
};
} // namespace Capsaicin
