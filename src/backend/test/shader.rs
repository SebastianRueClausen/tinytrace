use std::{borrow::Cow, mem};

use ash::vk;

use crate::backend::*;

fn create_storage_buffer(context: &mut Context) -> Handle<Buffer> {
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
}

#[test]
fn compute_shader() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();

    let a = create_storage_buffer(&mut context);
    let b = create_storage_buffer(&mut context);
    let c = create_storage_buffer(&mut context);

    let data: Box<[u8]> = (0..=255).collect();

    context
        .write_buffers(&[BufferWrite {
            buffer: a.clone(),
            data: Cow::Borrowed(&data),
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
                push_constant_size: None,
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
        .dispatch(256, 1)
        .unwrap();

    // `b` to `c`.
    context
        .bind_buffer("src", &b)
        .bind_buffer("dst", &c)
        .dispatch(256, 1)
        .unwrap();

    let download = context.download(&[c.clone()], &[]).unwrap();
    assert_eq!(data, download.buffers[&c]);
}

#[test]
fn push_constant() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let dst = create_storage_buffer(&mut context);
    let bindings = &[Binding {
        name: "dst",
        ty: BindingType::StorageBuffer {
            ty: "uint",
            array: true,
            writes: true,
        },
    }];
    let shader = context
        .create_shader(
            Lifetime::Static,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(256).height(1),
                source: include_str!("push_constant.glsl"),
                push_constant_size: Some(mem::size_of::<u32>() as u32),
                bindings,
                includes: &[],
            },
        )
        .unwrap();
    let download = context
        .bind_shader(&shader)
        .push_constant(&0xdeadbeef_u32)
        .bind_buffer("dst", &dst)
        .dispatch(256, 1)
        .unwrap()
        .download(&[dst.clone()], &[])
        .unwrap();
    let data: &[u32] = bytemuck::cast_slice(&download.buffers[&dst]);
    assert_eq!(data, &[0xdeadbeef_u32; 64]);
}
