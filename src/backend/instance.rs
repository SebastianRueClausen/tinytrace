use std::ffi::{self, CStr, CString};
use std::ops::Deref;

use crate::error::Error;
use ash::{ext, khr, vk};

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_utils: ext::debug_utils::Instance,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Instance {
    pub fn new(validate: bool) -> Result<Self, Error> {
        use vk::{
            DebugUtilsMessageSeverityFlagsEXT as Severity, DebugUtilsMessageTypeFlagsEXT as Type,
        };
        let entry = unsafe { ash::Entry::load()? };
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                Severity::ERROR | Severity::WARNING | Severity::INFO | Severity::VERBOSE,
            )
            .message_type(
                Type::GENERAL | Type::PERFORMANCE | Type::VALIDATION | Type::DEVICE_ADDRESS_BINDING,
            )
            .pfn_user_callback(Some(debug_callback));
        let layers = validate
            .then(|| vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()])
            .unwrap_or_default();
        let layer_names: Vec<_> = layers.iter().map(|layer| layer.as_ptr()).collect();
        let extension_names = [
            ext::debug_utils::NAME.as_ptr(),
            khr::surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::wayland_surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::xlib_surface::NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            khr::xcb_surface::NAME.as_ptr(),
        ];
        let application_info =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::default()
            .push_next(&mut debug_info)
            .application_info(&application_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_info, None)? };
        Ok(Self {
            entry,
            instance,
            debug_utils,
            debug_messenger,
        })
    }

    pub fn destroy(&self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut ffi::c_void,
) -> vk::Bool32 {
    let types = vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING;
    let severities = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    if types.contains(ty) && severities.contains(severity) {
        let message = CStr::from_ptr((*cb_data).p_message);
        if cfg!(test) {
            panic!("vulkan({ty:?}): {message:?}\n");
        } else {
            println!("vulkan({ty:?}): {message:?}\n");
        }
    }
    vk::FALSE
}
