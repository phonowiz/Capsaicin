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
        uint32_t nrc_train_active = 0;
        uint32_t nrc_inference_active = 1;
        float    nrc_learning_rate = 0.01f;
        uint32_t nrc_batch_size = 1024;
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

    GfxProgram nrc_inference_program_;
    GfxProgram nrc_train_program_;
    GfxKernel  inference_kernel_;
    GfxKernel  train_kernel_;
    GfxKernel  update_weights_kernel_; // Optional if integrated into train

    GfxTexture output_texture_; // Stores NRC inference result
    GfxBuffer  counters_buffer_; // [0] = Query Count, [1] = Sample Count

    // Compute Checks


    // RT Members
    GfxProgram rt_program_;
    GfxKernel  rt_kernel_;
    GfxSbt     rt_sbt_;
    GfxBuffer  ray_camera_buffer_;

    RenderOptions options;

    bool initKernels(CapsaicinInternal const &capsaicin) noexcept;
    void setupSbt(CapsaicinInternal const &capsaicin) const noexcept;
    void setupRTKernel(CapsaicinInternal const &capsaicin,
        std::vector<GfxLocalRootSignatureAssociation> &local_root_signature_associations,
        std::vector<std::string> &defines, std::vector<std::string> &exports,
        std::vector<std::string> &subobjects) noexcept;
};
} // namespace Capsaicin
