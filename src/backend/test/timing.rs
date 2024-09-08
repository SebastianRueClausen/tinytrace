use crate::backend::*;

#[test]
fn timestamps() {
    let mut context = Context::new(None).unwrap();
    let buffer = context
        .create_buffer(
            Lifetime::Static,
            &BufferRequest {
                size: 0xffffff,
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
            },
        )
        .unwrap();
    context
        .insert_timestamp("before")
        .fill_buffer(&buffer, 0)
        .unwrap()
        .insert_timestamp("after");
    context.wait_until_idle().unwrap();
    // Just check that results are available.
    let _before = context.timestamp("before").unwrap();
    let _after = context.timestamp("after").unwrap();
}
