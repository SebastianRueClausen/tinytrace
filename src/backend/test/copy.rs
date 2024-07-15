use ash::vk;

use super::create_test_buffer;
use crate::backend::*;

fn random_bytes(count: usize) -> Box<[u8]> {
    (0..count).map(|value| (value % 255) as u8).collect()
}

#[test]
fn transfer() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let buffer = create_test_buffer(&mut context, 1024);
    let extent = vk::Extent3D::default().width(32).height(32).depth(1);
    let image = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                format: vk::Format::R8G8B8A8_SRGB,
                memory_location: MemoryLocation::Device,
                mip_level_count: 1,
                extent,
            },
        )
        .unwrap();
    let buffer_data = random_bytes(context.buffer(&buffer).size as usize);
    let image_data = random_bytes((extent.width * extent.height * 4) as usize);
    let download = context
        .write_buffers(&[BufferWrite {
            buffer: buffer.clone(),
            data: &buffer_data,
        }])
        .unwrap()
        .write_images(&[ImageWrite {
            image: image.clone(),
            offset: vk::Offset3D::default(),
            extent,
            mips: &[image_data.clone()],
        }])
        .unwrap()
        .download(&[buffer.clone()], &[image.clone()])
        .unwrap();
    assert_eq!(download.buffers[&buffer], buffer_data);
    assert_eq!(download.images[&image], image_data);
}

#[test]
fn transfer_image_mips() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let extent = vk::Extent3D::default().width(32).height(32).depth(1);
    let image = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                format: vk::Format::R8G8B8A8_SRGB,
                memory_location: MemoryLocation::Device,
                mip_level_count: 2,
                extent,
            },
        )
        .unwrap();
    let mips: [_; 2] = [
        random_bytes((extent.width * extent.height * 4) as usize),
        random_bytes(((extent.width / 2) * (extent.height / 2) * 4) as usize),
    ];
    let download = context
        .write_images(&[ImageWrite {
            image: image.clone(),
            offset: vk::Offset3D::default(),
            extent,
            mips: &mips.clone(),
        }])
        .unwrap()
        .download(&[], &[image.clone()])
        .unwrap();
    // Check that the mips are packed tightly.
    assert_eq!(download.images[&image][..mips[0].len()], *mips[0]);
    assert_eq!(download.images[&image][mips[0].len()..], *mips[1]);
}

#[test]
fn odd_sized_images() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();

    let a_extent = vk::Extent3D::default().width(83).height(47).depth(1);
    let b_extent = vk::Extent3D::default().width(97).height(59).depth(1);

    let a = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                format: vk::Format::R8G8B8A8_SRGB,
                memory_location: MemoryLocation::Device,
                mip_level_count: 1,
                extent: a_extent,
            },
        )
        .unwrap();
    let b = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                format: vk::Format::R8G8B8A8_SRGB,
                memory_location: MemoryLocation::Device,
                mip_level_count: 1,
                extent: b_extent,
            },
        )
        .unwrap();

    let a_bytes = random_bytes((a_extent.width * a_extent.height * 4) as usize);
    let b_bytes = random_bytes((b_extent.width * b_extent.height * 4) as usize);

    let download = context
        .write_images(&[ImageWrite {
            image: a.clone(),
            offset: vk::Offset3D::default(),
            extent: a_extent,
            mips: &[a_bytes.clone()],
        }])
        .unwrap()
        .write_images(&[ImageWrite {
            image: b.clone(),
            offset: vk::Offset3D::default(),
            extent: b_extent,
            mips: &[b_bytes.clone()],
        }])
        .unwrap()
        .download(&[], &[a.clone(), b.clone()])
        .unwrap();
    // Check that the mips are packed tightly.
    assert_eq!(&download.images[&a], &a_bytes);
    assert_eq!(&download.images[&b], &b_bytes);
}

#[test]
fn odd_sized_buffers() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let a = create_test_buffer(&mut context, 7919);
    let b = create_test_buffer(&mut context, 7727);
    let a_data = random_bytes(context.buffer(&a).size as usize);
    let b_data = random_bytes(context.buffer(&b).size as usize);
    context
        .write_buffers(&[
            BufferWrite {
                buffer: a.clone(),
                data: &a_data,
            },
            BufferWrite {
                buffer: b.clone(),
                data: &b_data,
            },
        ])
        .unwrap();
    let download = context.download(&[a.clone(), b.clone()], &[]).unwrap();
    assert_eq!(download.buffers[&a], a_data);
    assert_eq!(download.buffers[&b], b_data);
}

#[test]
fn large_buffers() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let buffers: Vec<_> = (0..2)
        .map(|_| {
            context
                .create_buffer(
                    Lifetime::Frame,
                    &BufferRequest {
                        memory_location: MemoryLocation::Device,
                        size: 1024 * 1024 * 100,
                        ty: BufferType::Storage,
                    },
                )
                .unwrap()
        })
        .collect();
    let data_1 = vec![1; 1024].into_boxed_slice();
    let data_2 = vec![2; 1024].into_boxed_slice();
    context
        .write_buffers(&[
            BufferWrite {
                buffer: buffers[0].clone(),
                data: &data_1,
            },
            BufferWrite {
                buffer: buffers[1].clone(),
                data: &data_2,
            },
        ])
        .unwrap();
    let download = context.download(&buffers, &[]).unwrap();
    assert_eq!(download.buffers[&buffers[0]][..1024], *data_1);
    assert_eq!(download.buffers[&buffers[1]][..1024], *data_2);
}
