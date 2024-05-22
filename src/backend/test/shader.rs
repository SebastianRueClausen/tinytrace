use ash::vk;

use crate::backend::*;

#[test]
fn compute_shader() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();

    let mut create_storage_buffer = || {
        context
            .create_buffer(
                Lifetime::Static,
                &BufferRequest {
                    memory_location: MemoryLocation::Device,
                    ty: BufferType::Storage,
                    size: 256,
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

    let bindings = &[
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
    let shader = context
        .create_shader(
            Lifetime::Static,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(256).height(1),
                source: include_str!("copy_buffer.glsl"),
                bindings,
                includes: &[],
            },
        )
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
