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

#[test]
fn allocate_large_buffers() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    for _ in 0..2 {
        let _ = context
            .create_buffer(
                Lifetime::Frame,
                &BufferRequest {
                    memory_location: MemoryLocation::Device,
                    size: 1024 * 1024 * 100,
                    ty: BufferType::Storage,
                },
            )
            .unwrap();
    }
}
