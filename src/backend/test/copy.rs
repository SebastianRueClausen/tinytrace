use ash::vk;

use super::create_test_buffer;
use crate::backend::*;

#[test]
fn transfer() {
    let render_size = vk::Extent2D::default().width(1024).height(1024);
    let mut context = Context::new(None, render_size).unwrap();
    let buffer = create_test_buffer(&mut context);
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
    let buffer_data: Box<[u8]> = (0..=255).collect();
    let image_data: Box<[u8]> = (0..extent.width * extent.height * 4)
        .map(|value| (value % 255) as u8)
        .collect();
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
    let mips: [Box<[u8]>; 2] = [
        (0..extent.width * extent.height * 4).map(|_| 0u8).collect(),
        (0..(extent.width / 2) * (extent.height / 2) * 4)
            .map(|_| 1u8)
            .collect(),
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
