use std::fmt;

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
    pub light_sampling: LightSampling,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            light_sampling: LightSampling::default(),
            tonemap: true,
            bounce_count: 4,
            sample_count: 1,
        }
    }
}
