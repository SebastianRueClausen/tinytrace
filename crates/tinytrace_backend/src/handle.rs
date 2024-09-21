use super::Lifetime;
use std::{hash, marker};

/// The handle of a resource.
#[derive(Debug, Copy)]
pub struct Handle<T> {
    pub(super) lifetime: Lifetime,
    pub(super) index: usize,
    pub(super) epoch: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Handle<T> {
    pub(super) fn new(lifetime: Lifetime, epoch: usize, vec: &mut Vec<T>, value: T) -> Self {
        vec.push(value);
        Self {
            index: vec.len() - 1,
            epoch,
            lifetime,
            _marker: marker::PhantomData,
        }
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.epoch == other.epoch && self.lifetime == other.lifetime
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            lifetime: self.lifetime,
            index: self.index,
            epoch: self.epoch,
            _marker: marker::PhantomData,
        }
    }
}

impl<T> hash::Hash for Handle<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.lifetime.hash(state);
        self.index.hash(state);
        self.epoch.hash(state);
    }
}
