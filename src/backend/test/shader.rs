use ash::vk;

use crate::{backend::*, read_file};

#[test]
fn compute_shader() {
    let mut context = Context::new().unwrap();
    let mut create_storage_buffer = || {
        context
            .create_buffer(
                Lifetime::Static,
                &BufferRequest {
                    size: 256,
                    ty: BufferType::Storage,
                    memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                },
            )
            .unwrap()
    };

    let src = create_storage_buffer();
    let dst = create_storage_buffer();

    let data: Box<[u8]> = (0..=255).collect();

    context
        .write_buffers(&[BufferWrite {
            buffer: src.clone(),
            data: &data,
        }])
        .unwrap();

    let source = read_file!("copy_buffer.glsl");
    let bindings = [
        Binding {
            name: "src",
            ty: BindingType::StorageBuffer {
                ty: "int",
                array: true,
                writes: false,
            },
        },
        Binding {
            name: "dst",
            ty: BindingType::StorageBuffer {
                ty: "int",
                array: true,
                writes: true,
            },
        },
    ];
    let grid_size = vk::Extent2D::default().width(256).height(1);
    let shader = context.create_shader(source, grid_size, &bindings).unwrap();

    context.bind_shader(&shader);
    context.bind_buffer("src", &src);
    context.bind_buffer("dst", &dst);
    context.dispatch(256, 1);

    let download = context.download(&[dst.clone()], &[]).unwrap();
    assert_eq!(data, download.buffers[&dst]);
}
