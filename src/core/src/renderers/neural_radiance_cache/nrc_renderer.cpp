/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
********************************************************************/

#include "neural_radiance_cache/NeuralRadianceCache.h"
#include "renderer.h"
#include "tone_mapping/tone_mapping.h"
#include "auto_exposure/auto_exposure.h"

namespace Capsaicin
{
class NeuralRadianceCacheRenderer final
    : public Renderer
    , RendererFactory::Registrar<NeuralRadianceCacheRenderer>
{
public:
    static constexpr std::string_view Name = "Neural Radiance Cache";

    NeuralRadianceCacheRenderer() noexcept {}

    std::vector<std::unique_ptr<RenderTechnique>> setupRenderTechniques(
        [[maybe_unused]] RenderOptionList const &renderOptions) noexcept override
    {
        std::vector<std::unique_ptr<RenderTechnique>> render_techniques;
        render_techniques.emplace_back(std::make_unique<NeuralRadianceCache>());
        render_techniques.emplace_back(std::make_unique<AutoExposure>());
        render_techniques.emplace_back(std::make_unique<ToneMapping>()); // Basic tone mapping
        return render_techniques;
    }
};
} // namespace Capsaicin
