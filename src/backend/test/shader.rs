use ash::vk;

use crate::{backend::*, read_file};

#[test]
fn compute_shader() {
    let mut context = Context::new(None).unwrap();
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

    let a = create_storage_buffer();
    let b = create_storage_buffer();
    let c = create_storage_buffer();

    let data: Box<[u8]> = (0..=255).collect();

    context
        .write_buffers(&[BufferWrite {
            buffer: a.clone(),
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
    let shader = context
        .create_shader(source, grid_size, &bindings, &[])
        .unwrap();

    context.bind_shader(&shader);

    // `a` to `b`.
    context
        .bind_buffer("src", &a)
        .bind_buffer("dst", &b)
        .dispatch(256, 1);

    // `b` to `c`.
    context
        .bind_buffer("src", &b)
        .bind_buffer("dst", &c)
        .dispatch(256, 1);

    let download = context.download(&[c.clone()], &[]).unwrap();
    assert_eq!(data, download.buffers[&c]);
}
