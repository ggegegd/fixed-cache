//! A minimalistic one-way set associative cache, generic over key-value types.
//!
//! This cache has a fixed size to allow fast access and minimize per-call overhead.

#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::new_without_default)]

use core::{
    cell::UnsafeCell,
    hash::{BuildHasher, Hash},
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};
use equivalent::Equivalent;

const LOCKED_BIT: usize = 0x0000_8000;

#[cfg(feature = "rapidhash")]
type DefaultBuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
#[cfg(not(feature = "rapidhash"))]
type DefaultBuildHasher = std::hash::RandomState;

/// A concurrent set-associative cache, generic over key-value types.
///
/// The cache uses a fixed number of entries and resolves collisions by eviction.
/// It is designed for fast access with minimal per-call overhead.
///
/// # Type Parameters
///
/// - `K`: The key type, must implement `Hash + Eq`.
/// - `V`: The value type, must implement `Clone`.
/// - `S`: The hash builder type, must implement `BuildHasher`.
pub struct Cache<K, V, S = DefaultBuildHasher> {
    entries: *const [Bucket<(K, V)>],
    build_hasher: S,
    drop: bool,
}

impl<K, V, S> core::fmt::Debug for Cache<K, V, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cache").finish_non_exhaustive()
    }
}

// SAFETY: `Cache` is safe to share across threads because `Bucket` uses atomic operations.
unsafe impl<K: Send, V: Send, S: Send> Send for Cache<K, V, S> {}
unsafe impl<K: Send, V: Send, S: Sync> Sync for Cache<K, V, S> {}

impl<K, V, S> Cache<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Create a new cache with the specified number of entries and hasher.
    ///
    /// Dynamically allocates memory for the cache entries.
    ///
    /// # Panics
    ///
    /// Panics if `num` is not a power of two.
    pub fn new(num: usize, build_hasher: S) -> Self {
        let bytes = num * size_of::<Bucket<(K, V)>>();
        let raw = Box::into_raw(vec![0u8; bytes].into_boxed_slice());
        let cast = std::ptr::slice_from_raw_parts_mut(raw.cast(), num);
        Self::new_inner(cast, build_hasher, false)
    }

    /// Creates a new cache with the specified entries and hasher.
    ///
    /// # Panics
    ///
    /// Panics if `entries.len()` is not a power of two.
    #[inline]
    pub const fn new_static(entries: &'static [Bucket<(K, V)>], build_hasher: S) -> Self {
        Self::new_inner(entries, build_hasher, false)
    }

    #[inline]
    const fn new_inner(entries: *const [Bucket<(K, V)>], build_hasher: S, drop: bool) -> Self {
        assert!(entries.len().is_power_of_two());
        Self { entries, build_hasher, drop }
    }

    #[inline]
    const fn index_mask(&self) -> usize {
        let n = self.capacity();
        unsafe { core::hint::assert_unchecked(n.is_power_of_two()) };
        n - 1
    }

    #[inline]
    const fn tag_mask(&self) -> usize {
        !self.index_mask()
    }

    /// Returns the hash builder used by this cache.
    #[inline]
    pub const fn hasher(&self) -> &S {
        &self.build_hasher
    }

    /// Returns the number of entries in this cache.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.entries.len()
    }
}

impl<K, V, S> Cache<K, V, S>
where
    K: Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    /// Get an entry from the cache.
    pub fn get<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> Option<V> {
        let (bucket, tag) = self.calc(key);
        self.get_inner(key, bucket, tag)
    }

    #[inline]
    fn get_inner<Q: ?Sized + Hash + Equivalent<K>>(
        &self,
        key: &Q,
        bucket: &Bucket<(K, V)>,
        tag: usize,
    ) -> Option<V> {
        if bucket.try_lock(Some(tag)) {
            // SAFETY: We hold the lock, so we have exclusive access.
            let (ck, v) = unsafe { (*bucket.data.get()).assume_init_ref() };
            if key.equivalent(ck) {
                let v = v.clone();
                bucket.unlock(tag);
                return Some(v);
            }
            bucket.unlock(tag);
            // Hash collision: same hash but different key.
        }

        None
    }

    /// Insert an entry into the cache.
    pub fn insert(&self, key: K, value: V) {
        let (bucket, tag) = self.calc(&key);
        self.insert_inner(|| key, || value, bucket, tag);
    }

    #[inline]
    fn insert_inner(
        &self,
        make_key: impl FnOnce() -> K,
        make_value: impl FnOnce() -> V,
        bucket: &Bucket<(K, V)>,
        tag: usize,
    ) {
        if bucket.try_lock(None) {
            // SAFETY: We hold the lock, so we have exclusive access.
            unsafe {
                let data = (&mut *bucket.data.get()).as_mut_ptr();
                (&raw mut (*data).0).write(make_key());
                (&raw mut (*data).1).write(make_value());
            }
            bucket.unlock(tag);
        }
    }

    /// Gets a value from the cache, or inserts one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns a clone of the cached value.
    /// Otherwise, calls `f` to compute the value, attempts to insert it, and returns it.
    #[inline]
    pub fn get_or_insert_with<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce(&K) -> V,
    {
        let mut key = std::mem::ManuallyDrop::new(key);
        let mut read = false;
        let r = self.get_or_insert_with_ref(&*key, f, |k| {
            read = true;
            unsafe { std::ptr::read(k) }
        });
        if !read {
            unsafe { std::mem::ManuallyDrop::drop(&mut key) }
        }
        r
    }

    /// Gets a value from the cache, or inserts one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns a clone of the cached value.
    /// Otherwise, calls `f` to compute the value, attempts to insert it, and returns it.
    ///
    /// This is the same as [`get_or_insert_with`], but takes a reference to the key, and a function
    /// to get the key reference to an owned key.
    ///
    /// [`get_or_insert_with`]: Self::get_or_insert_with
    #[inline]
    pub fn get_or_insert_with_ref<Q, F, Cvt>(&self, key: &Q, f: F, cvt: Cvt) -> V
    where
        Q: ?Sized + Hash + Equivalent<K>,
        F: FnOnce(&Q) -> V,
        Cvt: FnOnce(&Q) -> K,
    {
        let (bucket, tag) = self.calc(key);
        if let Some(v) = self.get_inner(key, bucket, tag) {
            return v;
        }
        let value = f(key);
        self.insert_inner(|| cvt(key), || value.clone(), bucket, tag);
        value
    }

    #[inline]
    fn calc<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> (&Bucket<(K, V)>, usize) {
        let hash = self.hash_key(key);
        // SAFETY: index is masked to be within bounds.
        let bucket = unsafe { (&*self.entries).get_unchecked(hash & self.index_mask()) };
        let tag = hash & self.tag_mask();
        (bucket, tag)
    }

    #[inline]
    fn hash_key<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> usize {
        let hash = self.build_hasher.hash_one(key);

        if cfg!(target_pointer_width = "32") {
            ((hash >> 32) as usize) ^ (hash as usize)
        } else {
            hash as usize
        }
    }
}

impl<K, V, S> Drop for Cache<K, V, S> {
    fn drop(&mut self) {
        if self.drop {
            drop(unsafe { Box::from_raw(self.entries.cast_mut()) });
        }
    }
}

/// A cache bucket.
#[repr(C, align(128))]
pub struct Bucket<T> {
    tag: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Bucket<T> {
    /// Creates a new zeroed bucket.
    #[inline]
    pub const fn new() -> Self {
        Self { tag: AtomicUsize::new(0), data: UnsafeCell::new(MaybeUninit::zeroed()) }
    }

    #[inline]
    fn try_lock(&self, expected: Option<usize>) -> bool {
        let state = self.tag.load(Ordering::Relaxed);
        if let Some(expected) = expected {
            if state != expected {
                return false;
            }
        } else if state & LOCKED_BIT != 0 {
            return false;
        }
        self.tag
            .compare_exchange(state, state | LOCKED_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    #[inline]
    fn unlock(&self, tag: usize) {
        self.tag.store(tag, Ordering::Release);
    }
}

// SAFETY: `Bucket` is a specialized `Mutex<T>` that never blocks.
unsafe impl<T: Send> Send for Bucket<T> {}
unsafe impl<T: Send> Sync for Bucket<T> {}

/// Declares a static cache with the given name, key type, value type, and size.
///
/// The size must be a power of two.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rapidhash")] {
/// use fixed_cache::{Cache, static_cache};
///
/// type BuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
///
/// static MY_CACHE: Cache<u64, &'static str, BuildHasher> =
///     static_cache!(u64, &'static str, 1024, BuildHasher::new());
///
/// let value = MY_CACHE.get_or_insert_with(&42, |_k| "hi");
/// assert_eq!(value, "hi");
///
/// let new_value = MY_CACHE.get_or_insert_with(&42, |_k| "not hi");
/// assert_eq!(new_value, "not hi");
/// # }
/// ```
#[macro_export]
macro_rules! static_cache {
    ($K:ty, $V:ty, $size:expr) => {
        $crate::static_cache!($K, $V, $size, Default::default())
    };
    ($K:ty, $V:ty, $size:expr, $hasher:expr) => {{
        static ENTRIES: [$crate::Bucket<($K, $V)>; $size] =
            [const { $crate::Bucket::new() }; $size];
        $crate::Cache::new_static(&ENTRIES, $hasher)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn new_cache<K: Hash + Eq, V: Clone>(size: usize) -> Cache<K, V> {
        Cache::new(size, Default::default())
    }

    #[test]
    fn test_basic_get_or_insert() {
        let cache = new_cache(1024);

        let mut computed = false;
        let value = cache.get_or_insert_with(42, |&k| {
            computed = true;
            k * 2
        });
        assert!(computed);
        assert_eq!(value, 84);

        computed = false;
        let value = cache.get_or_insert_with(42, |&k| {
            computed = true;
            k * 2
        });
        assert!(!computed);
        assert_eq!(value, 84);
    }

    #[test]
    fn test_different_keys() {
        let cache: Cache<String, usize> = static_cache!(String, usize, 1024);

        let v1 = cache.get_or_insert_with("hello".to_string(), |s| s.len());
        let v2 = cache.get_or_insert_with("world!".to_string(), |s| s.len());

        assert_eq!(v1, 5);
        assert_eq!(v2, 6);
    }

    #[test]
    fn test_new_dynamic_allocation() {
        let cache: Cache<u32, u32> = new_cache(64);
        assert_eq!(cache.capacity(), 64);

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));
    }

    #[test]
    fn test_get_miss() {
        let cache = new_cache::<u64, u64>(64);
        assert_eq!(cache.get(&999), None);
    }

    #[test]
    fn test_insert_and_get() {
        let cache: Cache<u64, String> = new_cache(64);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.get(&3), Some("three".to_string()));
        assert_eq!(cache.get(&4), None);
    }

    #[test]
    fn test_insert_twice() {
        let cache = new_cache(64);

        cache.insert(42, 1);
        assert_eq!(cache.get(&42), Some(1));

        cache.insert(42, 2);
        let v = cache.get(&42);
        assert!(v == Some(1) || v == Some(2));
    }

    #[test]
    fn test_get_or_insert_with_ref() {
        let cache: Cache<String, usize> = new_cache(64);

        let key = "hello";
        let value = cache.get_or_insert_with_ref(key, |s| s.len(), |s| s.to_string());
        assert_eq!(value, 5);

        let value2 = cache.get_or_insert_with_ref(key, |_| 999, |s| s.to_string());
        assert_eq!(value2, 5);
    }

    #[test]
    fn test_get_or_insert_with_ref_different_keys() {
        let cache: Cache<String, usize> = new_cache(1024);

        let v1 = cache.get_or_insert_with_ref("foo", |s| s.len(), |s| s.to_string());
        let v2 = cache.get_or_insert_with_ref("barbaz", |s| s.len(), |s| s.to_string());

        assert_eq!(v1, 3);
        assert_eq!(v2, 6);
    }

    #[test]
    fn test_capacity() {
        let cache = new_cache::<u64, u64>(256);
        assert_eq!(cache.capacity(), 256);

        let cache2 = new_cache::<u64, u64>(128);
        assert_eq!(cache2.capacity(), 128);
    }

    #[test]
    fn test_hasher() {
        let cache = new_cache::<u64, u64>(64);
        let _ = cache.hasher();
    }

    #[test]
    fn test_debug_impl() {
        let cache = new_cache::<u64, u64>(64);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
    }

    #[test]
    fn test_bucket_new() {
        let bucket: Bucket<(u64, u64)> = Bucket::new();
        assert_eq!(bucket.tag.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_many_entries() {
        let cache = new_cache(1024);

        for i in 0..500 {
            cache.insert(i, i * 2);
        }

        let mut hits = 0;
        for i in 0..500 {
            if cache.get(&i) == Some(i * 2) {
                hits += 1;
            }
        }
        assert!(hits > 0);
    }

    #[test]
    fn test_string_keys() {
        let cache = new_cache(64);

        cache.insert("alpha".to_string(), 1);
        cache.insert("beta".to_string(), 2);
        cache.insert("gamma".to_string(), 3);

        assert_eq!(cache.get(&"alpha".to_string()), Some(1));
        assert_eq!(cache.get(&"beta".to_string()), Some(2));
        assert_eq!(cache.get(&"gamma".to_string()), Some(3));
    }

    #[test]
    fn test_zero_values() {
        let cache = new_cache(64);

        cache.insert(0, 0);
        assert_eq!(cache.get(&0), Some(0));

        cache.insert(1, 0);
        assert_eq!(cache.get(&1), Some(0));
    }

    #[test]
    fn test_clone_value() {
        #[derive(Clone, PartialEq, Debug)]
        struct MyValue(Vec<u8>);

        let cache: Cache<u64, MyValue> = new_cache(64);

        cache.insert(1, MyValue(vec![1, 2, 3]));
        let v = cache.get(&1);
        assert_eq!(v, Some(MyValue(vec![1, 2, 3])));
    }

    fn run_concurrent<F>(num_threads: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        thread::scope(|s| {
            for t in 0..num_threads {
                let f = &f;
                s.spawn(move || f(t));
            }
        });
    }

    #[test]
    fn test_concurrent_reads() {
        let cache = new_cache(1024);

        for i in 0..100 {
            cache.insert(i, i * 10);
        }

        run_concurrent(4, |_| {
            for i in 0..100 {
                let _ = cache.get(&i);
            }
        });
    }

    #[test]
    fn test_concurrent_writes() {
        let cache = new_cache(1024);

        run_concurrent(4, |t| {
            for i in 0..100 {
                cache.insert((t * 1000 + i) as u64, i as u64);
            }
        });
    }

    #[test]
    fn test_concurrent_read_write() {
        let cache = new_cache(256);

        run_concurrent(2, |t| {
            for i in 0..1000u64 {
                if t == 0 {
                    cache.insert(i % 100, i);
                } else {
                    let _ = cache.get(&(i % 100));
                }
            }
        });
    }

    #[test]
    fn test_concurrent_get_or_insert() {
        let cache = new_cache(1024);

        run_concurrent(8, |_| {
            for i in 0..100 {
                let _ = cache.get_or_insert_with(i, |&k| k * 2);
            }
        });

        for i in 0..100 {
            if let Some(v) = cache.get(&i) {
                assert_eq!(v, i * 2);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_non_power_of_two_panics() {
        let _ = new_cache::<u64, u64>(100);
    }

    #[test]
    fn test_power_of_two_sizes() {
        for shift in 1..10 {
            let size = 1 << shift;
            let cache = new_cache::<u64, u64>(size);
            assert_eq!(cache.capacity(), size);
        }
    }

    #[test]
    fn test_small_cache() {
        let cache = new_cache(2);
        assert_eq!(cache.capacity(), 2);

        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        let count = [1, 2, 3].iter().filter(|&&k| cache.get(&k).is_some()).count();
        assert!(count <= 2);
    }

    #[test]
    fn test_equivalent_key_lookup() {
        let cache = new_cache(64);

        cache.insert("hello".to_string(), 42);

        assert_eq!(cache.get(&"hello".to_string()), Some(42));
    }

    #[test]
    fn test_large_values() {
        let cache = new_cache(64);

        let large_value: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        cache.insert(1, large_value.clone());

        assert_eq!(cache.get(&1), Some(large_value));
    }

    #[test]
    fn test_get_or_insert_does_not_leak_key() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Hash, PartialEq, Eq)]
        struct DropKey(u64);

        impl Drop for DropKey {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        impl Clone for DropKey {
            fn clone(&self) -> Self {
                DropKey(self.0)
            }
        }

        {
            let cache: Cache<DropKey, u64> = new_cache(64);

            let _ = cache.get_or_insert_with(DropKey(1), |k| k.0 * 2);
            let _ = cache.get_or_insert_with(DropKey(1), |k| k.0 * 2);
        }

        assert!(DROP_COUNT.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<Cache<u64, u64>>();
        assert_sync::<Cache<u64, u64>>();
        assert_send::<Bucket<(u64, u64)>>();
        assert_sync::<Bucket<(u64, u64)>>();
    }
}
