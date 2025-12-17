use super::*;
use std::{collections::HashMap, sync::Mutex};

type BuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;

static STATS: KeccakCacheStats = KeccakCacheStats {
    hits: [const { AtomicUsize::new(0) }; MAX_INPUT_LEN + 1],
    misses: [const { AtomicUsize::new(0) }; MAX_INPUT_LEN + 1],
    out_of_range: Mutex::new(HashMap::with_hasher(BuildHasher::new())),
    collisions: Mutex::new(Vec::new()),
};

struct KeccakCacheStats {
    hits: [AtomicUsize; MAX_INPUT_LEN + 1],
    misses: [AtomicUsize; MAX_INPUT_LEN + 1],
    out_of_range: Mutex<HashMap<usize, usize, BuildHasher>>,
    collisions: Mutex<Vec<(String, String)>>,
}

#[inline]
pub(super) fn hit(len: usize) {
    if !ENABLE_STATS {
        return;
    }
    STATS.hits[len].fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(super) fn miss(len: usize) {
    if !ENABLE_STATS {
        return;
    }
    STATS.misses[len].fetch_add(1, Ordering::Relaxed);
}

#[inline(never)]
pub(super) fn out_of_range(len: usize) {
    if !ENABLE_STATS {
        return;
    }
    *STATS.out_of_range.lock().unwrap().entry(len).or_insert(0) += 1;
}

#[inline(never)]
pub(super) fn collision(input: &[u8], cached: &[u8]) {
    if !ENABLE_STATS {
        return;
    }
    let input_hex = crate::hex::encode(input);
    let cached_hex = crate::hex::encode(cached);
    STATS.collisions.lock().unwrap().push((input_hex, cached_hex));
}

#[doc(hidden)]
pub fn format() -> String {
    use core::fmt::Write;
    let mut out = String::new();

    if !ENABLE_STATS {
        out.push_str("keccak cache stats: DISABLED");
        return out;
    }

    let mut total_hits = 0usize;
    let mut total_misses = 0usize;
    let mut entries: Vec<(usize, usize, usize)> = Vec::new();
    for len in 0..=MAX_INPUT_LEN {
        let hits = STATS.hits[len].load(Ordering::Relaxed);
        let misses = STATS.misses[len].load(Ordering::Relaxed);
        if hits > 0 || misses > 0 {
            entries.push((len, hits, misses));
            total_hits += hits;
            total_misses += misses;
        }
    }
    for (&len, &misses) in STATS.out_of_range.lock().unwrap().iter() {
        entries.push((len, 0, misses));
        total_misses += misses;
    }
    entries.sort_by_key(|(len, _, _)| *len);

    writeln!(out, "keccak cache stats by length:").unwrap();
    writeln!(out, "{:>6} {:>12} {:>12} {:>8}", "len", "hits", "misses", "hit%").unwrap();
    for (len, hits, misses) in entries {
        let total = hits + misses;
        let hit_rate = (hits as f64 / total as f64) * 100.0;
        writeln!(out, "{len:>6} {hits:>12} {misses:>12} {hit_rate:>7.1}%").unwrap();
    }
    let total = total_hits + total_misses;
    if total > 0 {
        let hit_rate = (total_hits as f64 / total as f64) * 100.0;
        writeln!(out, "{:>6} {:>12} {:>12} {:>7.1}%", "all", total_hits, total_misses, hit_rate)
            .unwrap();
    }

    let collisions = STATS.collisions.lock().unwrap();
    if !collisions.is_empty() {
        writeln!(out, "\nhash collisions ({}):", collisions.len()).unwrap();
        for (input, cached) in collisions.iter() {
            writeln!(out, "  input:  0x{input}").unwrap();
            writeln!(out, "  cached: 0x{cached}").unwrap();
            writeln!(out).unwrap();
        }
    }

    out
}
