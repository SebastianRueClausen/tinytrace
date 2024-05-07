use std::ops;

use crate::error::Error;
use ash::vk;

use super::{device::Device, Context, Handle, Lifetime};

#[derive(Debug, Clone)]
pub enum BindingType {
    StorageBuffer {
        ty: &'static str,
        count: u32,
        writes: bool,
    },
    UniformBuffer {
        ty: &'static str,
    },
    AccelerationStructure,
    SampledImage {
        count: u32,
    },
    StorageImage {
        count: u32,
        writes: bool,
    },
}

impl BindingType {
    fn to_glsl(&self, name: &str, set: u32, index: u32) -> String {
        let classifier = |writes| if writes { "" } else { "readonly " };
        let brackets = |count| if count > 1 { "[]" } else { "" };
        match self {
            Self::StorageBuffer { ty, count, writes } => {
                let (brackets, classifier) = (brackets(*count), classifier(*writes));
                format!("{classifier}buffer Set{set}Binding{index} {{ {ty} {name}{brackets}; }};")
            }
            Self::UniformBuffer { ty } => {
                format!("uniform Set{set}Index{index} {{ {ty} {name}; }};")
            }
            Self::AccelerationStructure => {
                format!("uniform accelerationStructureEXT {name};")
            }
            Self::SampledImage { count } => {
                format!("uniform sampled2D {name}{};", brackets(*count))
            }
            Self::StorageImage { count, writes } => {
                let (brackets, classifier) = (brackets(*count), classifier(*writes));
                format!("{classifier} uniform image2D {name}{brackets};")
            }
        }
    }

    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
            Self::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
            Self::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Self::SampledImage { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::StorageImage { .. } => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    #[allow(dead_code)]
    fn descriptor_size(&self, device: &Device) -> usize {
        let props = &device.descriptor_buffer_properties;
        match self {
            Self::StorageBuffer { .. } => props.storage_buffer_descriptor_size,
            Self::UniformBuffer { .. } => props.uniform_buffer_descriptor_size,
            Self::AccelerationStructure => props.acceleration_structure_descriptor_size,
            Self::SampledImage { .. } => props.sampled_image_descriptor_size,
            Self::StorageImage { .. } => props.storage_image_descriptor_size,
        }
    }

    fn count(&self) -> u32 {
        match self {
            Self::StorageBuffer { count, .. }
            | Self::SampledImage { count }
            | Self::StorageImage { count, .. } => *count,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub name: &'static str,
    pub ty: BindingType,
}

impl Binding {
    pub fn to_glsl(&self, set: u32, index: u32) -> String {
        let ty = self.ty.to_glsl(self.name, set, index);
        format!("layout (set = {set}, binding = {index}) {ty}\n")
    }
}

#[derive(Debug)]
pub struct DescriptorLayout {
    layout: vk::DescriptorSetLayout,
    bindings: Vec<Binding>,
}

impl ops::Deref for DescriptorLayout {
    type Target = vk::DescriptorSetLayout;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl DescriptorLayout {
    pub fn new(device: &Device, bindings: &[Binding]) -> Result<Self, Error> {
        let flags: Vec<_> = bindings
            .iter()
            .map(|binding| {
                (binding.ty.count() > 1)
                    .then_some(vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                    .unwrap_or_default()
            })
            .collect();
        let layout_bindings: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(location, binding)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(location as u32)
                    .descriptor_type(binding.ty.descriptor_type())
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_count(binding.ty.count())
            })
            .collect();
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .bindings(&layout_bindings)
            .push_next(&mut binding_flags);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        Ok(Self {
            layout,
            bindings: bindings.to_vec(),
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_descriptor_set_layout(self.layout, None) }
    }

    pub fn to_glsl(&self, set: u32) -> String {
        let to_glsl = |(index, binding): (usize, &Binding)| binding.to_glsl(set, index as u32);
        self.bindings.iter().enumerate().map(to_glsl).collect()
    }

    pub fn size(&self, device: &Device) -> vk::DeviceSize {
        let function = &device.descriptor_buffer;
        unsafe { function.get_descriptor_set_layout_size(self.layout) }
    }
}

impl Context {
    pub fn create_descriptor_layout(
        &mut self,
        bindings: &[Binding],
    ) -> Result<Handle<DescriptorLayout>, Error> {
        let layout = DescriptorLayout::new(&self.device, bindings)?;
        let pool = self.pool_mut(Lifetime::Static);
        Ok(Handle::new(
            Lifetime::Static,
            pool.epoch,
            &mut pool.descriptor_layouts,
            layout,
        ))
    }
}

#[test]
fn to_glsl() {
    let binding = Binding {
        name: "test",
        ty: BindingType::StorageBuffer {
            ty: "Test",
            count: 4,
            writes: true,
        },
    };
    let glsl = binding.to_glsl(0, 0);
    let expected = "layout (set = 0, binding = 0) buffer Set0Binding0 { Test test[]; };\n";
    assert_eq!(expected, glsl);
}
