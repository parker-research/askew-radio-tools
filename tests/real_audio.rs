//! Integration tests against real SatNOGS observation audio.
//!
//! These download real recordings over HTTPS on every `cargo test` run (not
//! `#[ignore]`d — this project isn't considered working until this passes
//! against real captures, so it runs by default). Requires network access.
//!
//! Downloaded files are cached under `target/test-cache/` (gitignored via
//! `/target`) so repeated runs don't re-fetch them. Each recording's SHA-256
//! is pinned, so an upstream change to (or corruption of) a recording fails
//! loudly instead of showing up as a mysterious decode regression.

use std::io::Write;
use std::path::{Path, PathBuf};

use askew_radio_tools::dsp;
use askew_radio_tools::pipeline::{self, FrameTier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

fn cache_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-cache");
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Download `url` into the cache dir (if not already present with the
/// expected contents) and return the local path. Panics if the downloaded
/// file's SHA-256 doesn't match `expected_sha256`.
fn fetch_cached(url: &str, expected_sha256: &str) -> PathBuf {
    let filename = url.rsplit('/').next().expect("url has a filename");
    let path = cache_dir().join(filename);

    // A cached file with the wrong hash (e.g. from an older download) is
    // re-fetched rather than trusted.
    if let Ok(cached) = std::fs::read(&path)
        && sha256_hex(&cached) == expected_sha256
    {
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

    let actual_sha256 = sha256_hex(&bytes);
    assert_eq!(
        actual_sha256, expected_sha256,
        "{url}: downloaded file's SHA-256 doesn't match the pinned hash — the \
         recording changed upstream or the download was corrupted"
    );

    let tmp_path = path.with_extension("part");
    std::fs::File::create(&tmp_path)
        .and_then(|mut f| f.write_all(&bytes))
        .unwrap_or_else(|e| panic!("failed to write {tmp_path:?}: {e}"));
    std::fs::rename(&tmp_path, &path).expect("finalize downloaded file");

    path
}

/// A "good" frame's fields, minus `filename` (which is just the input
/// path, not a property of the decode) — what we lock in exactly below.
#[derive(Debug, Clone, PartialEq)]
struct GoodFrame {
    time_in_file_ms: f64,
    data_length_bytes: usize,
    rs_corrected_error_count: Option<u32>,
    crc_pass: Option<bool>,
    data_hex: String,
}

/// One line of a `real_audio_fixtures/*.ndjson` file: the fields of a
/// verified frame that are pinned — the same names as in the decoder's
/// JSONL output, so a fixture is that output narrowed to these fields (see
/// [`assert_pinned_good_frames`] for the command).
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedFrame {
    time_in_file_ms: f64,
    data_length_bytes: usize,
    rs_corrected_error_count: u32,
    data_hex: String,
}

/// Parse a fixture file into `GoodFrame`s. All the frames these SatNOGS
/// observations decode to are FrontierSat CSP frames, which always carry a
/// valid CRC32C trailer (see `fec::csp_crc32c_check`'s doc comment) — so
/// every "good" (RS-correctable) frame here is expected to have
/// `crc_pass: Some(true)`.
fn good_frames(ndjson: &str) -> Vec<GoodFrame> {
    ndjson
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let frame: PinnedFrame = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("fixture line {}: {e}: {line}", i + 1));
            GoodFrame {
                time_in_file_ms: frame.time_in_file_ms,
                data_length_bytes: frame.data_length_bytes,
                rs_corrected_error_count: Some(frame.rs_corrected_error_count),
                crc_pass: Some(true),
                data_hex: frame.data_hex,
            }
        })
        .collect()
}

/// Render `frame` the way it's written in the fixture files, so a decoded
/// frame list can be pasted straight back into one. Frames that don't fit
/// the fixture format (missing RS count, or CRC not passing) fall back to
/// their `Debug` form.
fn fixture_line(frame: &GoodFrame) -> String {
    match (frame.rs_corrected_error_count, frame.crc_pass) {
        (Some(rs_corrected_error_count), Some(true)) => serde_json::to_string(&PinnedFrame {
            time_in_file_ms: frame.time_in_file_ms,
            data_length_bytes: frame.data_length_bytes,
            rs_corrected_error_count,
            data_hex: frame.data_hex.clone(),
        })
        .expect("a PinnedFrame always serializes"),
        _ => format!("{frame:?}"),
    }
}

fn fixture_lines(frames: &[GoodFrame]) -> String {
    frames
        .iter()
        .map(|f| format!("{}\n", fixture_line(f)))
        .collect()
}

/// Describe how `actual` differs from `expected`: frame counts, which
/// expected frames weren't decoded, which decoded frames weren't expected,
/// and then both lists in full (in fixture form).
fn describe_frame_mismatch(expected: &[GoodFrame], actual: &[GoodFrame]) -> String {
    let missing: Vec<GoodFrame> = expected
        .iter()
        .filter(|f| !actual.contains(f))
        .cloned()
        .collect();
    let unexpected: Vec<GoodFrame> = actual
        .iter()
        .filter(|f| !expected.contains(f))
        .cloned()
        .collect();

    format!(
        "expected {} verified frame(s), decoded {}\n\
         \n\
         {} expected frame(s) not decoded (or decoded differently):\n{}\
         \n\
         {} decoded frame(s) not expected:\n{}\
         \n\
         full expected list:\n{}\
         \n\
         full decoded list:\n{}",
        expected.len(),
        actual.len(),
        missing.len(),
        fixture_lines(&missing),
        unexpected.len(),
        fixture_lines(&unexpected),
        fixture_lines(expected),
        fixture_lines(actual),
    )
}

/// A SatNOGS observation pinned below: where to fetch its audio, the
/// recording's SHA-256, and the verified frames the decoder recovers from
/// it — kept in `tests/real_audio_fixtures/<observation id>.ndjson`, one
/// [`PinnedFrame`] per line, `frame_count` of them.
struct Observation {
    url: &'static str,
    sha256: &'static str,
    frames_ndjson: &'static str,
    frame_count: usize,
}

impl Observation {
    fn fetch(&self) -> PathBuf {
        fetch_cached(self.url, self.sha256)
    }
}

/// Download `observation` (verifying its SHA-256), decode it, and assert
/// that its [`FrameTier::Verified`] frames exactly match its fixture file —
/// pinning the whole DSP -> framing -> Golay -> RS -> CRC pipeline's current
/// decoding and timestamp precision against real hardware output, not just
/// synthetic fixtures.
///
/// Also asserts that the tiers below it are populated the way real,
/// noisy audio populates them: at least one [`FrameTier::Believable`]
/// frame (so the believability gate isn't so tight that it takes genuine
/// partially-corrupt bursts with it) and at least one
/// [`FrameTier::Candidate`] (so the pipeline is still *labelling* weak
/// candidates rather than quietly dropping them, which is what makes
/// `--output-filter all` a usable diagnostic).
///
/// If a pipeline change deliberately alters decoding or timestamp
/// precision, regenerate the relevant fixture with:
///
/// ```sh
/// cargo run --release -- --output-filter verified target/test-cache/satnogs_<id>_<time>.ogg \
///     | jq -c '{time_in_file_ms, data_length_bytes, rs_corrected_error_count, data_hex}' \
///     > tests/real_audio_fixtures/<id>.ndjson
/// ```
///
/// update the observation's `frame_count` to match, and eyeball the diff
/// before committing it — the payloads all decode to plausible FrontierSat
/// telemetry text, so a change here is either a real precision/behavior
/// change worth reviewing, or a regression.
fn assert_pinned_good_frames(observation: &Observation) {
    let path = observation.fetch();
    let path_str = path.to_str().expect("cache path is valid UTF-8");

    let records = pipeline::decode_file(path_str, dsp::DEFAULT_SYMBOL_RATE_HZ)
        .expect("pipeline should run without error");
    let expected: &[GoodFrame] = &good_frames(observation.frames_ndjson);
    assert_eq!(
        expected.len(),
        observation.frame_count,
        "{path_str}: the fixture file doesn't hold the observation's frame_count frames"
    );
    let good: Vec<GoodFrame> = records
        .iter()
        .filter(|r| r.tier == FrameTier::Verified)
        .map(|r| GoodFrame {
            time_in_file_ms: r.time_in_file_ms,
            data_length_bytes: r.data_length_bytes,
            rs_corrected_error_count: r.rs_corrected_error_count,
            crc_pass: r.crc_pass,
            data_hex: r.data_hex.clone(),
        })
        .collect();

    let count_at = |tier| records.iter().filter(|r| r.tier == tier).count();
    let believable = count_at(FrameTier::Believable);
    let candidates = count_at(FrameTier::Candidate);
    eprintln!(
        "{path_str}: decoded {} frame(s) total: {} verified (expected {}), {} rs-correctable, \
         {believable} believable, {candidates} candidate",
        records.len(),
        good.len(),
        expected.len(),
        count_at(FrameTier::RsCorrectableCrcError),
    );

    assert!(
        good == expected,
        "{path_str}: decoded 'good' frames no longer match the pinned expectation — if this \
         is a deliberate algorithm/precision change, regenerate the fixture (see \
         assert_pinned_good_frames' doc comment) and review the diff before updating it\n\
         \n\
         {}",
        describe_frame_mismatch(expected, &good)
    );

    assert!(
        believable > 0,
        "{path_str}: expected at least one believable-but-RS-uncorrectable frame (this capture \
         is known to contain real bursts too corrupt for RS to fix, which the tier's syncword \
         + Golay header checks should still recognise as real)"
    );

    assert!(
        candidates > 0,
        "{path_str}: expected the noise-tier candidates to still be reported — the pipeline \
         labels weak frames rather than dropping them, and --output-filter all depends on it"
    );
}

const OBS_14813295: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/8/18/17/14813295/satnogs_14813295_2026-08-18T17-05-35.ogg",
    sha256: "c57f7313c1707b1afb9e23b9d7c53fa414f04c5334c50ffd9fbc0e5a5cad3b4b",
    frames_ndjson: include_str!("real_audio_fixtures/14813295.ndjson"),
    frame_count: 26,
};

#[test]
fn test_satnogs_observation_14813295_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_14813295);
}

const OBS_14183111: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/5/28/17/14183111/satnogs_14183111_2026-05-28T17-32-37.ogg",
    sha256: "e7b028917323808fdeaf0c178512579afc0093bced63047f7e73916daf39c16d",
    frames_ndjson: include_str!("real_audio_fixtures/14183111.ndjson"),
    frame_count: 19,
};

#[test]
fn test_satnogs_observation_14183111_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_14183111);
}

const OBS_15035794: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/22/22/15035794/satnogs_15035794_2026-09-22T22-45-21.ogg",
    sha256: "41a927f92e1b7b4268655be0c06a0a6d052f493fae59709766c8f9ba73e6d6cf",
    frames_ndjson: include_str!("real_audio_fixtures/15035794.ndjson"),
    frame_count: 7,
};

#[test]
fn test_satnogs_observation_15035794_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15035794);
}

const OBS_15035805: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/22/22/15035805/satnogs_15035805_2026-09-22T22-46-33.ogg",
    sha256: "1176cb38c48fd607d27564eea68b99fe61d9d95ce3502e46cfee2540fa46cdcd",
    frames_ndjson: include_str!("real_audio_fixtures/15035805.ndjson"),
    frame_count: 9,
};

#[test]
fn test_satnogs_observation_15035805_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15035805);
}

const OBS_15035811: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/22/22/15035811/satnogs_15035811_2026-09-22T22-47-00.ogg",
    sha256: "97d1ab0f098cad3fe28209617b2ee97fec724bc938e65d49a7c9306f43b41674",
    frames_ndjson: include_str!("real_audio_fixtures/15035811.ndjson"),
    frame_count: 3,
};

#[test]
fn test_satnogs_observation_15035811_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15035811);
}

const OBS_15035900: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/22/22/15035900/satnogs_15035900_2026-09-22T22-48-58.ogg",
    sha256: "5ee5eb53b5a996733ddde7191e1f549dd348f581547ee23c27deb3c1f6a5e888",
    frames_ndjson: include_str!("real_audio_fixtures/15035900.ndjson"),
    frame_count: 8,
};

#[test]
fn test_satnogs_observation_15035900_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15035900);
}

const OBS_15039637: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/22/22/15039637/satnogs_15039637_2026-09-22T22-46-31.ogg",
    sha256: "a2b6cd9fec5741eeec75c1a3146fb4a8b4f8fa4e07d84bb0b293d7ecaaaca13f",
    frames_ndjson: include_str!("real_audio_fixtures/15039637.ndjson"),
    frame_count: 138,
};

#[test]
fn test_satnogs_observation_15039637_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15039637);
}

const OBS_15039753: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/11/15039753/satnogs_15039753_2026-09-23T11-03-05.ogg",
    sha256: "f46322f7a47f78754b3e59092d5d20667c13b5fce66783dafd353a6dd6163a38",
    frames_ndjson: include_str!("real_audio_fixtures/15039753.ndjson"),
    frame_count: 382,
};

/// Observation 15039753 contains serveral segments of back-to-back messages in a bulk file downlink.
#[test]
fn test_satnogs_observation_15039753_decodes_exact_good_frames() {
    // TODO: Expected to contain at least 415 real packets. This pin is against regressions; hopefully many more packets one day.
    assert_pinned_good_frames(&OBS_15039753);
}

const OBS_15040978: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/12/15040978/satnogs_15040978_2026-09-23T12-05-33.ogg",
    sha256: "fabc3018ffb3e95f13b4d813b80e34e5f31a96d95284de58ae63a4f176f02375",
    frames_ndjson: include_str!("real_audio_fixtures/15040978.ndjson"),
    frame_count: 42,
};

#[test]
fn test_satnogs_observation_15040978_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15040978);
}

const OBS_15040999: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/14/15040999/satnogs_15040999_2026-09-23T14-37-10.ogg",
    sha256: "c617f0819a5e15663a7188795ad4cd8e3d438139ba9b74ee5854c1dcaa89d51d",
    frames_ndjson: include_str!("real_audio_fixtures/15040999.ndjson"),
    frame_count: 30,
};

#[test]
fn test_satnogs_observation_15040999_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15040999);
}

const OBS_15041834: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/20/15041834/satnogs_15041834_2026-09-23T20-52-44.ogg",
    sha256: "cb0182d63f033b9ede09e21704a56ccb7c9e8904a5cf29d6b1a618221c0934a5",
    frames_ndjson: include_str!("real_audio_fixtures/15041834.ndjson"),
    frame_count: 349,
};

#[test]
fn test_satnogs_observation_15041834_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15041834);
}

const OBS_15041859: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/20/15041859/satnogs_15041859_2026-09-23T20-54-50.ogg",
    sha256: "8dc151870965771a5a9f0844ccb7b2e41e858427054eeb3170a4ecb2ede563b6",
    frames_ndjson: include_str!("real_audio_fixtures/15041859.ndjson"),
    frame_count: 4,
};

#[test]
fn test_satnogs_observation_15041859_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15041859);
}

const OBS_15041863: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/23/18/15041863/satnogs_15041863_2026-09-23T18-18-19.ogg",
    sha256: "e422b2d67b329dcfbae24d5b21777451ac9cd0dd681549f3962310fe31ba50cb",
    frames_ndjson: include_str!("real_audio_fixtures/15041863.ndjson"),
    frame_count: 0,
};

#[test]
fn test_satnogs_observation_15041863_decodes_exact_good_frames() {
    // The pipeline currently recovers no verified frames from this capture —
    // pinned empty so any new verified frame (true or false positive) gets reviewed.
    assert_pinned_good_frames(&OBS_15041863);
}

const OBS_15046793: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/24/16/15046793/satnogs_15046793_2026-09-24T16-59-39.ogg",
    sha256: "b44f28f72787f106bc3d7a7221ab881043a75b77ced80b3e74fc42ebb89be849",
    frames_ndjson: include_str!("real_audio_fixtures/15046793.ndjson"),
    frame_count: 374,
};

#[test]
fn test_satnogs_observation_15046793_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15046793);
}

const OBS_15046795: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/24/16/15046795/satnogs_15046795_2026-09-24T16-58-16.ogg",
    sha256: "beb741209f6ac785e8deaef7d8b410d6ddad44de80de8e851b7d2504e567d12b",
    frames_ndjson: include_str!("real_audio_fixtures/15046795.ndjson"),
    frame_count: 163,
};

#[test]
fn test_satnogs_observation_15046795_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15046795);
}

const OBS_15046840: Observation = Observation {
    url: "https://network-satnogs.freetls.fastly.net/media/data_obs/2026/9/24/16/15046840/satnogs_15046840_2026-09-24T16-25-23.ogg",
    sha256: "e19c1c25b0fcc34513ac66e8de353816207d1137f5a939b0e8c1788114f0bed0",
    frames_ndjson: include_str!("real_audio_fixtures/15046840.ndjson"),
    frame_count: 27,
};

#[test]
fn test_satnogs_observation_15046840_decodes_exact_good_frames() {
    assert_pinned_good_frames(&OBS_15046840);
}
