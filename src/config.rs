use std::fmt;

/// How to sample scatter rays.
#[repr(u32)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Uniformly sample the surface hemisphere. This works resonably well for diffuse
    /// surfaces but will cause specular surfaces to appear dark unless many samples are used.
    UniformHemisphere = 1,
    /// Cosine weighted importance sampling of the surface hemisphere. This slightly decreases
    /// the noise of diffuse surfaces, but specular surfaces will still appear dark unless many
    /// samples are used.
    CosineHemisphere = 2,
    /// Multiple importance sample the surface.
    /// The surface metallic property is used as a balance heuristic to choose between GGX and
    /// cosine weigthed importance sampling. GGX importance sampling randomly samples a
    /// microsurface normal based on the GGX microfacet distribution. This works well for both
    /// specular and diffuse surfaces.
    #[default]
    Brdf = 3,
}

impl fmt::Display for SampleStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UniformHemisphere => write!(f, "Uniform Hemisphere"),
            Self::CosineHemisphere => write!(f, "Cosine Hemisphere"),
            Self::Brdf => write!(f, "BRDF"),
        }
    }
}

/// How to gather light along a path.
#[repr(u32)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LightSampling {
    /// The only light contribution is from emissive surfaces randomly hit by paths.
    OnHit = 1,
    /// Direct light is explicitly sampled along a path at each bounce if possible.
    #[default]
    NextEventEstimation = 2,
}

impl fmt::Display for LightSampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OnHit => write!(f, "None"),
            Self::NextEventEstimation => write!(f, "Next event estimation"),
        }
    }
}

/// Configuration of the renderer.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Config {
    /// The number of ray bounces.
    pub bounce_count: u32,
    /// The number of samples per pixel.
    pub sample_count: u32,
    pub tonemap: bool,
    pub sample_strategy: SampleStrategy,
    pub light_sampling: LightSampling,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_strategy: SampleStrategy::default(),
            light_sampling: LightSampling::default(),
            tonemap: true,
            bounce_count: 4,
            sample_count: 1,
        }
    }
}
