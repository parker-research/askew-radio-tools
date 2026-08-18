//! Integration tests against real SatNOGS observation audio.
//!
//! These download real recordings over HTTPS on every `cargo test` run (not
//! `#[ignore]`d — this project isn't considered working until this passes
//! against real captures, so it runs by default). Requires network access.
//!
//! Downloaded files are cached under `target/test-cache/` (gitignored via
//! `/target`) so repeated runs don't re-fetch them.

use std::io::Write;
use std::path::{Path, PathBuf};

use ax100_radio_csp_decoder::pipeline;

fn cache_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-cache");
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

/// Download `url` into the cache dir (if not already present) and return
/// the local path.
fn fetch_cached(url: &str) -> PathBuf {
    let filename = url.rsplit('/').next().expect("url has a filename");
    let path = cache_dir().join(filename);

    if path.exists() {
        return path;
    }

    let response = ureq::get(url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("failed to read response body from {url}: {e}"));

    let tmp_path = path.with_extension("part");
    std::fs::File::create(&tmp_path)
        .and_then(|mut f| f.write_all(&bytes))
        .unwrap_or_else(|e| panic!("failed to write {tmp_path:?}: {e}"));
    std::fs::rename(&tmp_path, &path).expect("finalize downloaded file");

    path
}

/// A real 9600 baud AX100 Mode 5 SatNOGS observation, known to contain at
/// least 10 valid (RS + CSP CRC passing) frames — used as a smoke test for
/// the whole DSP -> framing -> RS -> CRC pipeline against real hardware
/// output, not just synthetic fixtures.
#[test]
fn test_satnogs_observation_14813295_decodes_at_least_10_good_frames() {
    let url = "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/8/18/17/14813295/satnogs_14813295_2026-08-18T17-05-35.ogg";
    let path = fetch_cached(url);
    let path_str = path.to_str().expect("cache path is valid UTF-8");

    let records = pipeline::decode_file(path_str).expect("pipeline should run without error");

    let good: Vec<_> = records.iter().filter(|r| r.crc_pass).collect();
    let bad_count = records.len() - good.len();

    eprintln!(
        "decoded {} frame(s) total: {} good (CRC pass), {} bad (CRC fail)",
        records.len(),
        good.len(),
        bad_count
    );
    for r in &good {
        eprintln!(
            "  t={:>9.1}ms  {}B  rs_corrected={}  {}",
            r.time_in_file_ms, r.data_length_bytes, r.rs_corrected_error_count, r.data_hex
        );
    }

    assert!(
        good.len() >= 10,
        "expected at least 10 good (RS + CRC passing) frames, got {} (out of {} total)",
        good.len(),
        records.len()
    );
}
