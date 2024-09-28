# Tinytrace
A tiny realtime ray tracer.
The goal is to write a fairly capable ray tracer in 5000-7000 lines of code by utilizing hardware ray tracing and modern techniques such as ReSTIR.

## Example
To run example GLTF scene viewer, run:
```bash
cargo run --release --example viewer
```

## Vulkan
The ray tracer uses vulkan to accelerate the ray tracing. To make the implementation simpler, tinytrace uses a few fairly new Vulkan extensions, although they should be supported on almost all GPUs that support hardware ray tracing.

### Required Extensions

* VK_EXT_descriptor_buffer ([blog](https://www.khronos.org/blog/vk-ext-descriptor-buffer))
* VK_KHR_acceleration_structure ([blog](https://www.khronos.org/blog/ray-tracing-in-vulkan#Acceleration_Structures))
* VK_KHR_deferred_host_operations ([blog](https://www.khronos.org/blog/ray-tracing-in-vulkan#blog_Deferred_Operations))
* VK_KHR_ray_query ([blog](https://www.khronos.org/blog/ray-tracing-in-vulkan#blog_Ray_Queries))
* VK_KHR_ray_tracing_position_fetch ([blog](https://www.khronos.org/blog/introducing-vulkan-ray-tracing-position-fetch-extension))
* VK_KHR_spirv_1_4

### Required Features
The following are the non-universally-supported features required.

**Vulkan 1.0:**
* shaderInt16

**Vulkan 1.1:**
* storageBuffer16BitAccess
* uniformAndStorageBuffer16BitAccess

**Vulkan 1.2:**
* bufferDeviceAddress
* descriptorBufferBindingVariableDescriptorCount
* runtimeDescriptorArray
* shaderFloat16
* timelineSemaphore

**Descriptor Buffer:**
* DescriptorBufferImageLayoutIgnored
