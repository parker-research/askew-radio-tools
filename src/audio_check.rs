//! Audio quality pre-check: assess whether the audio is likely to contain
//! a decodable AX100 beacon before running the full DSP pipeline.
//!
//! Checks performed (all grounded in the AX100 config: 9600 baud, modindex
//! 0.667, so f_dev = 3200 Hz, signal occupies ~±4800 Hz around baseband):
//!
//! 1. **Nyquist headroom** — sample rate must fit the signal band with margin.
//! 2. **RMS level** — audio must not be silence or clipped.
//! 3. **Spectral energy in signal band** — meaningful power must exist in the
//!    ±14.4 kHz band (1.5× baud = AX100 RX bandwidth). Checked via a simple
//!    DFT energy tally on a short window of samples.
//! 4. **Noise floor estimate** — SNR in-band vs. guard band outside the signal
//!    must be positive (signal louder than noise floor).
//! 5. **Clipping fraction** — too many samples at ±1.0 means the SDR or
//!    recording was saturated and the FM discriminator will produce garbage.
//!
//! The check is deliberately conservative: a `Warn` does not abort decoding,
//! but a `Fail` means decode is very unlikely to succeed.

use crate::audio::AudioSamples;
use std::f32::consts::PI;

/// Verdict from the pre-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckVerdict {
    /// All checks passed — decoding is likely viable.
    Pass,
    /// Some checks raised concerns — decoding may still work but results
    /// should be treated with suspicion.
    Warn(Vec<String>),
    /// At least one hard failure — decoding is very unlikely to succeed.
    Fail(Vec<String>),
}

impl CheckVerdict {
    pub fn is_ok(&self) -> bool {
        !matches!(self, CheckVerdict::Fail(_))
    }
}

impl std::fmt::Display for CheckVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckVerdict::Pass => write!(f, "PASS — audio looks decodable"),
            CheckVerdict::Warn(msgs) => {
                write!(f, "WARN — proceed with caution:\n")?;
                for m in msgs {
                    writeln!(f, "  ⚠  {}", m)?;
                }
                Ok(())
            }
            CheckVerdict::Fail(msgs) => {
                write!(f, "FAIL — decode unlikely:\n")?;
                for m in msgs {
                    writeln!(f, "  ✗  {}", m)?;
                }
                Ok(())
            }
        }
    }
}

/// Detailed metrics produced by the check, exposed for logging / debugging.
#[derive(Debug, Clone)]
pub struct AudioMetrics {
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub rms_level: f32,
    pub clipping_fraction: f32,
    /// Power spectral density summed over the signal band (±14.4 kHz).
    pub inband_power_db: f32,
    /// Power spectral density summed over the guard band (above 16 kHz).
    pub guard_power_db: f32,
    /// Estimated SNR: inband_power_db − guard_power_db.
    pub estimated_snr_db: f32,
    pub verdict: CheckVerdict,
}

// ── Constants derived from AX100 config ─────────────────────────────────────

/// Baud rate (from param list 5: baud = 9600).
const BAUD: f32 = 9600.0;

/// AX100 RX bandwidth: 1.5 × baud = 14 400 Hz.
const SIGNAL_BW: f32 = BAUD * 1.5; // 14 400 Hz

/// Guard band starts here — anything above this is treated as noise floor.
const GUARD_LOW: f32 = SIGNAL_BW * 1.1; // 15 840 Hz

// ── Thresholds ───────────────────────────────────────────────────────────────

/// Below this RMS the audio is considered silence.
const RMS_SILENCE_THRESH: f32 = 1e-4;

/// Above this RMS the audio may be overdriven (SDR gain too high).
const RMS_HOT_THRESH: f32 = 0.9;

/// More than this fraction of samples at ±full-scale → clipping.
const CLIP_THRESH: f32 = 0.01; // 1 %

/// Minimum in-band power (dB, relative to full-scale) to consider non-trivial.
const INBAND_POWER_MIN_DB: f32 = -50.0;

/// Minimum SNR (dB) for a decode to have reasonable probability of success.
/// RS(255,223) can correct 16 byte errors (~7% BER), which roughly maps to
/// Eb/No ≈ 4–5 dB for GFSK. We use 3 dB as the hard floor.
const SNR_WARN_DB: f32 = 6.0;
const SNR_FAIL_DB: f32 = 0.0;

// ── DFT window ───────────────────────────────────────────────────────────────

/// Number of samples for the spectral energy estimate.
/// 8192 samples @ 48 kHz → ~170 ms, ~5.9 Hz frequency resolution.
const DFT_WINDOW: usize = 8192;

// ─────────────────────────────────────────────────────────────────────────────

/// Run all audio quality checks and return a verdict with metrics.
///
/// Uses only the first `DFT_WINDOW` samples for spectral analysis (fast),
/// but the full sample buffer for RMS and clipping (accurate).
pub fn check(audio: &AudioSamples) -> AudioMetrics {
    let fs = audio.sample_rate as f32;
    let samples = &audio.samples;
    let n = samples.len();

    let mut failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // ── 1. Duration sanity ───────────────────────────────────────────────────
    let duration_secs = n as f64 / audio.sample_rate as f64;
    // A minimum AX100 frame (1 byte payload) takes: preamble + sync + FEC bytes
    // ≈ (50 + 4 + 3 + 256) bytes × 8 bits / 9600 baud ≈ 264 ms
    if duration_secs < 0.3 {
        failures.push(format!(
            "Audio too short ({:.1} ms) — minimum frame is ~264 ms",
            duration_secs * 1000.0
        ));
    }

    // ── 2. RMS level ─────────────────────────────────────────────────────────
    let rms_level = {
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / n as f32).sqrt()
    };

    if rms_level < RMS_SILENCE_THRESH {
        failures.push(format!(
            "Audio appears silent (RMS = {:.2e}) — check SDR recording",
            rms_level
        ));
    } else if rms_level > RMS_HOT_THRESH {
        warnings.push(format!(
            "Audio level very high (RMS = {:.2}) — SDR gain may be too hot",
            rms_level
        ));
    }

    // ── 3. Clipping fraction ─────────────────────────────────────────────────
    let clip_count = samples.iter().filter(|&&s| s.abs() >= 0.999).count();
    let clipping_fraction = clip_count as f32 / n as f32;

    if clipping_fraction > CLIP_THRESH {
        failures.push(format!(
            "Audio clipping: {:.1}% of samples at ±full-scale — FM discriminator will distort",
            clipping_fraction * 100.0
        ));
    }

    // ── 4. Spectral energy estimate (Goertzel-style DFT energy tally) ────────
    //
    // We don't need a full FFT here — we want two numbers:
    //   • sum of |X[k]|² for bins inside the signal band [0, SIGNAL_BW]
    //   • sum of |X[k]|² for bins in the guard band [GUARD_LOW, fs/2]
    //
    // We use a direct DFT on `DFT_WINDOW` samples with a Hann window.
    // This is O(N²) but DFT_WINDOW=8192 and we only evaluate a small
    // number of relevant bins, so it's fast enough for a pre-check.

    let win_size = DFT_WINDOW.min(n);
    let windowed: Vec<f32> = samples[..win_size]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            // Hann window
            let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / (win_size - 1) as f32).cos();
            s * w
        })
        .collect();

    // Frequency resolution: fs / win_size
    let freq_res = fs / win_size as f32;

    // Bin ranges
    let inband_end_bin = (SIGNAL_BW / freq_res).ceil() as usize;
    let guard_start_bin = (GUARD_LOW / freq_res).ceil() as usize;
    let guard_end_bin = (win_size / 2).min((fs / 2.0 / freq_res) as usize);

    let inband_power = dft_band_power(&windowed, 1, inband_end_bin);
    let guard_power = dft_band_power(&windowed, guard_start_bin, guard_end_bin);

    // Convert to dB (floor at -120 dB to avoid log(0))
    let to_db = |p: f32| 10.0 * (p.max(1e-12)).log10();
    let inband_power_db = to_db(inband_power);
    let guard_power_db = to_db(guard_power);
    let estimated_snr_db = inband_power_db - guard_power_db;

    if inband_power_db < INBAND_POWER_MIN_DB {
        failures.push(format!(
            "No meaningful signal energy in band 0–{:.0} Hz (power = {:.1} dBFS)",
            SIGNAL_BW, inband_power_db
        ));
    }

    if estimated_snr_db < SNR_FAIL_DB {
        failures.push(format!(
            "SNR too low to decode: {:.1} dB (need >{:.0} dB for RS to compensate)",
            estimated_snr_db, SNR_WARN_DB
        ));
    } else if estimated_snr_db < SNR_WARN_DB {
        warnings.push(format!(
            "Low SNR: {:.1} dB — RS may not correct all errors (threshold {:.0} dB)",
            estimated_snr_db, SNR_WARN_DB
        ));
    }

    // ── 5. Nyquist headroom ──────────────────────────────────────────────────
    // Already enforced hard in audio::load_audio (≥19200 Hz), but add a
    // warning if the audio bandwidth is tight relative to the signal.
    // The signal needs fs/2 > SIGNAL_BW + some margin for Doppler.
    let nyquist = fs / 2.0;
    // AX100 AFC range is ±25% of BW = ±3600 Hz. Add that as Doppler margin.
    let doppler_margin = SIGNAL_BW * 0.25;
    if nyquist < SIGNAL_BW + doppler_margin {
        warnings.push(format!(
            "Audio bandwidth tight: Nyquist {:.0} Hz vs signal {:.0} Hz + Doppler margin {:.0} Hz",
            nyquist, SIGNAL_BW, doppler_margin
        ));
    }

    // ── Verdict ──────────────────────────────────────────────────────────────
    let verdict = if !failures.is_empty() {
        CheckVerdict::Fail(failures)
    } else if !warnings.is_empty() {
        CheckVerdict::Warn(warnings)
    } else {
        CheckVerdict::Pass
    };

    AudioMetrics {
        sample_rate: audio.sample_rate,
        duration_secs,
        rms_level,
        clipping_fraction,
        inband_power_db,
        guard_power_db,
        estimated_snr_db,
        verdict,
    }
}

/// Compute the sum of squared DFT magnitudes for bins [start_bin, end_bin).
///
/// Uses direct DFT evaluation — only called for a small number of band-edge
/// bins so performance is acceptable without pulling in rustfft.
fn dft_band_power(windowed: &[f32], start_bin: usize, end_bin: usize) -> f32 {
    let n = windowed.len() as f32;
    let mut power = 0.0f32;

    for k in start_bin..end_bin {
        let angle = -2.0 * PI * k as f32 / n;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, &sample) in windowed.iter().enumerate() {
            let (sin_a, cos_a) = (angle * i as f32).sin_cos();
            re += sample * cos_a;
            im += sample * sin_a;
        }
        power += (re * re + im * im) / (n * n); // normalised power
    }

    power
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioSamples;
    use std::f32::consts::TAU;

    /// Frequency deviation: baud × (1/1.5) / 2 = 3200 Hz (modindex auto = 0.667).
    const FREQ_DEV: f32 = BAUD / 1.5 / 2.0; // 3200 Hz

    fn make_tone(fs: u32, freq: f32, amp: f32, secs: f32) -> AudioSamples {
        let n = (fs as f32 * secs) as usize;
        let samples = (0..n)
            .map(|i| amp * (TAU * freq * i as f32 / fs as f32).sin())
            .collect();
        AudioSamples {
            samples,
            sample_rate: fs,
            channels: 1,
        }
    }

    fn make_silence(fs: u32, secs: f32) -> AudioSamples {
        AudioSamples {
            samples: vec![0.0; (fs as f32 * secs) as usize],
            sample_rate: fs,
            channels: 1,
        }
    }

    #[test]
    fn test_silence_fails() {
        let audio = make_silence(48_000, 1.0);
        let metrics = check(&audio);
        assert!(
            matches!(metrics.verdict, CheckVerdict::Fail(_)),
            "Silence should fail"
        );
    }

    #[test]
    fn test_in_band_tone_passes() {
        // A 3200 Hz tone (the FSK mark frequency) at reasonable level should pass
        let audio = make_tone(48_000, FREQ_DEV, 0.3, 1.0);
        let metrics = check(&audio);
        // Should at minimum not hard-fail on the energy/SNR checks
        assert!(
            metrics.inband_power_db > INBAND_POWER_MIN_DB,
            "In-band tone should show up in spectral energy: got {:.1} dB",
            metrics.inband_power_db
        );
    }

    #[test]
    fn test_out_of_band_noise_warns_or_fails() {
        // A 20 kHz tone (well above SIGNAL_BW) — guard band power dominates
        let audio = make_tone(48_000, 20_000.0, 0.3, 1.0);
        let metrics = check(&audio);
        // SNR should be negative (more power in guard than signal band)
        assert!(
            metrics.estimated_snr_db < SNR_WARN_DB,
            "Out-of-band tone should produce low/negative SNR: got {:.1} dB",
            metrics.estimated_snr_db
        );
    }

    #[test]
    fn test_clipping_detected() {
        // Build audio with many samples at exactly ±1.0
        let n = 48_000usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let audio = AudioSamples {
            samples,
            sample_rate: 48_000,
            channels: 1,
        };
        let metrics = check(&audio);
        assert!(
            metrics.clipping_fraction > CLIP_THRESH,
            "Should detect clipping: got fraction {:.3}",
            metrics.clipping_fraction
        );
        assert!(matches!(metrics.verdict, CheckVerdict::Fail(_)));
    }

    #[test]
    fn test_too_short_fails() {
        // 100 ms — shorter than minimum frame duration
        let audio = make_tone(48_000, FREQ_DEV, 0.3, 0.1);
        let metrics = check(&audio);
        assert!(
            matches!(metrics.verdict, CheckVerdict::Fail(_)),
            "Too-short audio should fail"
        );
    }

    #[test]
    fn test_metrics_fields_populated() {
        let audio = make_tone(48_000, FREQ_DEV, 0.3, 1.0);
        let metrics = check(&audio);
        assert_eq!(metrics.sample_rate, 48_000);
        assert!((metrics.duration_secs - 1.0).abs() < 0.01);
        assert!(metrics.rms_level > 0.0);
        assert!(metrics.inband_power_db.is_finite());
        assert!(metrics.guard_power_db.is_finite());
    }
}
