use std::time::Duration;

use ash::vk;

use super::{device::Device, Context, Error};

#[derive(Debug)]
pub(super) struct QueryPool {
    pub pool: vk::QueryPool,
    pub timestamps: Vec<String>,
}

impl QueryPool {
    pub const QUERY_COUNT: u32 = 2048;

    pub fn new(device: &Device) -> Result<Self, Error> {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(Self::QUERY_COUNT);
        Ok(Self {
            pool: unsafe { device.create_query_pool(&create_info, None)? },
            timestamps: Vec::new(),
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_query_pool(self.pool, None);
        }
    }

    pub fn add_timestamp(&mut self, name: String) -> u32 {
        assert!(
            self.timestamps.len() < QueryPool::QUERY_COUNT as usize,
            "too many queries"
        );
        self.timestamps.push(name);
        self.timestamps.len() as u32 - 1
    }

    pub fn get_queries(&mut self, device: &Device) -> Result<Vec<(String, Duration)>, Error> {
        if self.timestamps.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = vec![0u64; self.timestamps.len()];
        unsafe {
            let flags = vk::QueryResultFlags::TYPE_64;
            device.get_query_pool_results(self.pool, 0, &mut results, flags)?;
        }
        Ok(results
            .into_iter()
            .zip(self.timestamps.drain(..))
            .map(|(timestamp, name)| {
                let nanos = timestamp as f64 * device.properties.limits.timestamp_period as f64;
                (name, Duration::from_nanos(nanos as u64))
            })
            .collect())
    }
}

impl Context {
    pub fn insert_timestamp(&mut self, name: impl Into<String>) -> &mut Self {
        self.command_buffers
            .first_mut()
            .unwrap()
            .add_timestamp(&self.device, name.into());
        self
    }
}
