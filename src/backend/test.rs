use ash::vk;

use super::*;

fn create_test_buffer(context: &mut Context) -> Handle<Buffer> {
    context
        .create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                usage_flags: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                size: 256,
            },
        )
        .unwrap()
}

#[test]
fn transfer() {
    let mut context = Context::new().unwrap();
    let buffer = create_test_buffer(&mut context);
    let extent = vk::Extent3D::default().width(32).height(32).depth(1);
    let image = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                extent,
                format: vk::Format::R8G8B8A8_SRGB,
                usage_flags: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                mip_level_count: 1,
            },
        )
        .unwrap();
    let buffer_data: Box<[u8]> = (0..=255).collect();
    let image_data: Box<[u8]> = (0..extent.width * extent.height * 4)
        .map(|value| (value % 255) as u8)
        .collect();
    context
        .write_buffers(&[BufferWrite {
            buffer: buffer.clone(),
            data: &buffer_data,
        }])
        .unwrap();
    context
        .write_images(&[ImageWrite {
            image: image.clone(),
            offset: vk::Offset3D::default(),
            extent,
            mips: &[image_data.clone()],
        }])
        .unwrap();
    let download = context
        .download(&[buffer.clone()], &[image.clone()])
        .unwrap();
    assert_eq!(download.buffers[&buffer], buffer_data);
    assert_eq!(download.images[&image], image_data);
}

#[test]
fn transfer_image_mips() {
    let mut context = Context::new().unwrap();
    let extent = vk::Extent3D::default().width(32).height(32).depth(1);
    let image = context
        .create_image(
            Lifetime::Frame,
            &ImageRequest {
                extent,
                format: vk::Format::R8G8B8A8_SRGB,
                usage_flags: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                mip_level_count: 2,
            },
        )
        .unwrap();
    let mips: [Box<[u8]>; 2] = [
        (0..extent.width * extent.height * 4).map(|_| 0u8).collect(),
        (0..(extent.width / 2) * (extent.height / 2) * 4)
            .map(|_| 1u8)
            .collect(),
    ];
    context
        .write_images(&[ImageWrite {
            image: image.clone(),
            offset: vk::Offset3D::default(),
            extent,
            mips: &mips.clone(),
        }])
        .unwrap();
    let download = context.download(&[], &[image.clone()]).unwrap();
    // Check that the mips are packed tightly.
    assert_eq!(download.images[&image][..mips[0].len()], *mips[0]);
    assert_eq!(download.images[&image][mips[0].len()..], *mips[1]);
}

#[test]
fn compute_shader() {
    let mut context = Context::new().unwrap();
    let a = create_test_buffer(&mut context);
    let b = create_test_buffer(&mut context);
    let c = create_test_buffer(&mut context);
    let source = "void main() { }";
    let bindings = [
        Binding {
            name: "a",
            ty: BindingType::UniformBuffer { ty: "int" },
        },
        Binding {
            name: "b",
            ty: BindingType::UniformBuffer { ty: "int" },
        },
    ];
    let grid_size = vk::Extent2D::default().width(32).height(32);
    let shader = context.create_shader(source, grid_size, &bindings).unwrap();

    context.bind_shader(&shader);
    // First dispatch.
    context.bind_buffer("a", &a);
    context.bind_buffer("b", &b);
    context.dispatch(128, 128);
    // Second dispatch.
    context.bind_buffer("b", &c);
    context.dispatch(128, 128);

    context.execute_commands().unwrap();
}
