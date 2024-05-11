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
