mod copy;
mod shader;

use crate::backend::*;

fn create_test_buffer(context: &mut Context, size: vk::DeviceSize) -> Handle<Buffer> {
    context
        .create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                ty: BufferType::Uniform,
                memory_location: MemoryLocation::Device,
                size,
            },
        )
        .unwrap()
}
