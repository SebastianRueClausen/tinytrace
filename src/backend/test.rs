use ash::vk;

use super::*;

#[test]
fn transfer_buffer() {
    let mut context = Context::new().unwrap();
    let buffer = context
        .create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                usage_flags: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                size: 256,
            },
        )
        .unwrap();
    let data: Box<[u8]> = (0..=255).collect();
    context
        .write_buffers(&[BufferWrite {
            buffer: buffer.clone(),
            data: &data,
        }])
        .unwrap();
    let download = context.download(&[buffer.clone()], &[]).unwrap();
    assert_eq!(download.buffers[&buffer], data);
}

#[test]
fn transfer_image() {
    let mut context = Context::new().unwrap();
    let extent = vk::Extent3D::default().width(32).height(32).depth(1);
    let image = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                extent,
                format: vk::Format::R8G8B8A8_SRGB,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                mip_level_count: 1,
            },
        )
        .unwrap();
    let data: Box<[u8]> = (0..extent.width * extent.height * 4)
        .map(|value| (value % 255) as u8)
        .collect();
    context
        .write_images(&[ImageWrite {
            image: image.clone(),
            offset: vk::Offset3D::default(),
            extent,
            mips: &[data.clone()],
        }])
        .unwrap();
    let download = context.download(&[], &[image.clone()]).unwrap();
    assert_eq!(download.images[&image], data);
}
