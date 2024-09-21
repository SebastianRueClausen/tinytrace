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

/// How paths are replayed in world space ReSTIR.
#[repr(u32)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RestirReplay {
    /// Upon sampling from a reservoir, it's assumed that it's possible to reconnect to the
    /// first bounce and that the path remains valid. This can cause a lot of bias even in static
    /// scenes.
    None = 1,
    /// Upon sampling from a reservoir, a ray is traced from the sample point to the first bounce.
    /// This eliminates bias in static scenes, but may still cause bias if objects or lights are
    /// moved.
    #[default]
    First = 2,
    /// Retrace the full path to eleminate bias even in dynamic scenes.
    Full = 3,
}

impl fmt::Display for RestirReplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Configuration for the world space ReSTIR system.
#[derive(Debug, Clone, Copy)]
pub struct RestirConfig {
    pub enabled: bool,
    pub scene_scale: f32,
    pub update_hash_grid_capacity: u32,
    pub reservoir_hash_grid_capacity: u32,
    pub updates_per_cell: u32,
    pub reservoirs_per_cell: u32,
    pub replay: RestirReplay,
}

impl PartialEq for RestirConfig {
    fn eq(&self, other: &Self) -> bool {
        self.enabled == other.enabled
            && self.update_hash_grid_capacity == other.update_hash_grid_capacity
            && self.reservoir_hash_grid_capacity == other.reservoir_hash_grid_capacity
            && self.updates_per_cell == other.updates_per_cell
            && self.reservoirs_per_cell == other.reservoirs_per_cell
            && self.replay == other.replay
            && (self.scene_scale - other.scene_scale).abs() < 1e-4
    }
}

impl Eq for RestirConfig {}

impl Default for RestirConfig {
    fn default() -> Self {
        Self {
            replay: RestirReplay::default(),
            update_hash_grid_capacity: 0xffff,
            reservoir_hash_grid_capacity: 0xffff,
            updates_per_cell: 128,
            reservoirs_per_cell: 64,
            enabled: true,
            scene_scale: 10.0,
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
    pub restir: RestirConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            restir: RestirConfig::default(),
            sample_strategy: SampleStrategy::default(),
            light_sampling: LightSampling::default(),
            tonemap: true,
            bounce_count: 4,
            sample_count: 1,
        }
    }
}
