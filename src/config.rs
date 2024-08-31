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

/// Configuration for the world space ReSTIR system.
#[derive(Debug, Clone, Copy)]
pub struct RestirConfig {
    pub enabled: bool,
    pub scene_scale: f32,
    pub update_hash_grid_capacity: u32,
    pub reservoir_hash_grid_capacity: u32,
    pub updates_per_cell: u32,
    pub reservoirs_per_cell: u32,
}

impl PartialEq for RestirConfig {
    fn eq(&self, other: &Self) -> bool {
        self.enabled == other.enabled
            && self.update_hash_grid_capacity == other.update_hash_grid_capacity
            && self.reservoir_hash_grid_capacity == other.reservoir_hash_grid_capacity
            && self.updates_per_cell == other.updates_per_cell
            && self.reservoirs_per_cell == other.reservoirs_per_cell
            && (self.scene_scale - other.scene_scale).abs() < 1e-4
    }
}

impl Eq for RestirConfig {}

impl Default for RestirConfig {
    fn default() -> Self {
        Self {
            update_hash_grid_capacity: 0xffff,
            reservoir_hash_grid_capacity: 0xfffff,
            updates_per_cell: 128,
            reservoirs_per_cell: 8,
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
    pub restir: RestirConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            restir: RestirConfig::default(),
            sample_strategy: SampleStrategy::default(),
            tonemap: true,
            bounce_count: 6,
            sample_count: 2,
        }
    }
}
