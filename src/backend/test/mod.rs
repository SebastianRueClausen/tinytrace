mod copy;
mod shader;

use crate::backend::*;

fn create_test_buffer(context: &mut Context) -> Handle<Buffer> {
    context
        .create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                ty: BufferType::Uniform,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                size: 256,
            },
        )
        .unwrap()
}

#[macro_export]
macro_rules! read_file {
    ($fname:expr) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/backend/test/",
            $fname
        ))
    };
}

#[test]
fn allocate_large_buffers() {
    let mut context = Context::new().unwrap();
    for _ in 0..2 {
        let _ = context
            .create_buffer(
                Lifetime::Frame,
                &BufferRequest {
                    size: 1024 * 1024 * 100,
                    ty: BufferType::Storage,
                    memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                },
            )
            .unwrap();
    }
}
