//! DSP pipeline: low-pass filter → symbol timing → bit decisions.
//!
//! ## Pipeline overview
//!
//! ```text
//! f32 audio samples (already FM/GFSK-demodulated frequency deviation,
//!                     as produced by the SDR/ground-station chain)
//!      │
//!      ▼
//! [Biquad Low-Pass]    ~14.4 kHz cutoff (1.5× symbol rate)
//!      │
//!      ▼
//! [Gardner TED]        symbol timing recovery loop
//!      │
//!      ▼
//! [Slicer]             threshold at 0 → Vec<bool> NRZ bits
//! ```
//!
//! SatNOGS "audio" captures for AX100 are the *already demodulated*
//! instantaneous-frequency waveform (this is what gr-satellites'
//! `ax100_deframer` itself expects as input: "a float stream of soft
//! symbols" — the demod, typically `quadrature_demod_cf`, has already run
//! upstream in the SDR chain). So there is no FM/Hilbert discriminator
//! stage here: `audio.samples` is used directly as the frequency-deviation
//! signal, positive values meaning the "mark" tone and negative values the
//! "space" tone.
//!
//! (An earlier version of this module additionally ran a Hilbert-transform
//! based FM discriminator on top of this signal, which is both redundant
//! with an already-demodulated input and numerically wrong at AX100's
//! symbol rate: a 64-tap Hilbert FIR is not accurate enough at 3200 Hz
//! relative to a 48 kHz sample rate, and was verified to produce the same
//! sign for both +3200 Hz and -3200 Hz tones.)

use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type};

use crate::audio::AudioSamples;

/// Output of the DSP front-end: a stream of NRZ bits and the recovered
/// symbol rate (should be ≈ 9600 Hz).
pub struct BitStream {
    /// NRZ bits, MSB first, as recovered by the symbol timing loop.
    pub bits: Vec<bool>,
    /// Timestamp of each bit in `bits` (same index), in milliseconds from
    /// the start of the input audio file.
    pub bit_times_ms: Vec<f64>,
    /// Estimated symbol rate after timing recovery (Hz). Should be ≈ 9600.
    pub recovered_symbol_rate: f64,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full DSP front-end on a decoded audio buffer.
///
/// `samples` must be mono f32 normalised to [-1.0, 1.0] at a sample rate
/// high enough to represent 9600 baud (≥ 19200 Hz, typically 48000 Hz).
pub fn fm_discriminate_and_filter(audio: &AudioSamples) -> BitStream {
    let fs = audio.sample_rate as f64;
    let symbol_rate = 9600.0_f64;

    // 1. Low-pass filter at 1.5 × symbol_rate to strip out-of-band noise
    let lpf_cutoff = symbol_rate * 1.5; // 14 400 Hz
    let filtered = lowpass_filter(&audio.samples, fs, lpf_cutoff);

    // 2. Gardner timing error detector + interpolated sampling
    let (symbols, sample_positions, recovered_rate) = gardner_ted(&filtered, fs, symbol_rate);

    // 3. Hard decision (slicer): positive deviation → 1, negative → 0
    let bits: Vec<bool> = symbols.iter().map(|&s| s >= 0.0).collect();
    let bit_times_ms: Vec<f64> = sample_positions.iter().map(|&p| p / fs * 1000.0).collect();

    BitStream {
        bits,
        bit_times_ms,
        recovered_symbol_rate: recovered_rate,
    }
}

// ---------------------------------------------------------------------------
// Step 1: Biquad low-pass filter
// ---------------------------------------------------------------------------
//
// Second-order Butterworth IIR in Direct Form II Transposed.
// `biquad` crate handles the coefficient computation.

fn lowpass_filter(input: &[f32], fs: f64, cutoff_hz: f64) -> Vec<f32> {
    let coeffs = Coefficients::<f32>::from_params(
        Type::LowPass,
        (fs as f32).hz(),
        (cutoff_hz as f32).hz(),
        biquad::Q_BUTTERWORTH_F32,
    )
    .expect("LPF coefficient computation failed — check sample rate and cutoff values");

    let mut filter = DirectForm2Transposed::<f32>::new(coeffs);
    input.iter().map(|&s| filter.run(s)).collect()
}

// ---------------------------------------------------------------------------
// Step 2: Gardner timing error detector
// ---------------------------------------------------------------------------
//
// The Gardner TED is a decision-directed timing error detector that works
// on passband signals without a separate carrier reference. It estimates
// the timing error τ from mid-symbol and on-symbol samples:
//
//   e[k] = (y[k - T/2] - y[k + T/2]) · y[k]
//
// where T is the symbol period in samples, y[k] is the on-symbol sample,
// and y[k ± T/2] are the mid-symbol (strobe) samples.
//
// The loop filter is a simple PI controller:
//   μ[k+1] = μ[k] + α·e[k] + β·∑e[k]
//
// Typical values for α and β at 9600 baud / 48 kHz: α≈0.01, β≈0.001.

fn gardner_ted(input: &[f32], fs: f64, symbol_rate: f64) -> (Vec<f32>, Vec<f64>, f64) {
    let sps = fs / symbol_rate; // samples per symbol (e.g. 5.0 for 48000/9600)

    // The loop filter gains below are tuned assuming a roughly unit-
    // amplitude signal. `input` here is a frequency-deviation waveform
    // that can be at any scale (e.g. ±3200 for a 3200 Hz deviation), which
    // would otherwise blow the TED error — and thus the correction — up by
    // orders of magnitude and make the loop diverge instead of track. So
    // normalise by RMS first; this doesn't affect the final sign-based
    // slicer decision.
    let rms = (input.iter().map(|&x| x * x).sum::<f32>() / input.len().max(1) as f32).sqrt();
    let norm = if rms > 1e-12 { rms } else { 1.0 };
    let input: Vec<f32> = input.iter().map(|&x| x / norm).collect();
    let input = input.as_slice();

    // Loop filter constants (normalised loop bandwidth ≈ 0.01)
    let alpha = 0.01_f64; // proportional gain
    let beta = 0.001_f64; // integral gain

    let mut symbols: Vec<f32> = Vec::with_capacity(input.len() / sps as usize);
    let mut mu = 0.0_f64; // fractional timing offset [0, 1)
    let mut int_err = 0.0_f64; // integrator state

    // Track the actual sampling positions to estimate recovered symbol rate
    let mut last_sample_pos = 0.0_f64;
    let mut sample_positions: Vec<f64> = Vec::new();

    // We need the previous and previous-mid samples for the TED error.
    // Step through the input one symbol at a time.
    let mut idx = sps; // current (nominal) symbol strobe position

    while idx + sps < input.len() as f64 {
        // Integer and fractional parts of the strobe position
        let i_int = idx.floor() as usize;
        let i_frac = idx - idx.floor();

        // Linear interpolation helper
        let interp = |pos: f64| -> f32 {
            let i = pos.floor() as usize;
            let frac = (pos - pos.floor()) as f32;
            if i + 1 < input.len() {
                input[i] + frac * (input[i + 1] - input[i])
            } else {
                input[i.min(input.len() - 1)]
            }
        };

        // On-symbol sample y[k]
        let y_on = interp(idx);

        // Mid-symbol samples y[k - T/2] and y[k - 3T/2] (previous strobe)
        let y_mid_prev = interp(idx - sps / 2.0);
        let y_prev = interp(idx - sps);

        // Gardner TED error: uses *previous* symbol to avoid decision feedback
        let error = ((y_prev - y_on) * y_mid_prev) as f64;

        // PI loop filter. Clamp the integrator and the resulting correction
        // so a run of same-signed errors (e.g. a long string of identical
        // bits, or a sharp-edged square-wave deviation signal) can't wind
        // the loop up to the point where the strobe stalls or goes
        // backwards — which would hang the `while` loop below.
        int_err = (int_err + beta * error).clamp(-sps / 4.0, sps / 4.0);
        let correction = (alpha * error + int_err).clamp(-sps / 2.0, sps / 2.0);

        symbols.push(y_on);
        sample_positions.push(idx);

        // Advance strobe by one symbol period, adjusted by loop
        idx += sps - correction;
        let _ = i_int; // suppress unused warning
        let _ = i_frac;
    }

    // Estimate recovered symbol rate from mean inter-symbol spacing
    let recovered_rate = if sample_positions.len() > 1 {
        let mean_sps = (sample_positions.last().unwrap() - sample_positions[0])
            / (sample_positions.len() - 1) as f64;
        fs / mean_sps
    } else {
        symbol_rate
    };

    (symbols, sample_positions, recovered_rate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioSamples;

    /// Build a synthetic "already-demodulated" AX100 signal: a real-valued
    /// frequency-deviation waveform (like the output of an SDR's
    /// `quadrature_demod` block) that swings towards ±`freq_dev` for each
    /// bit. AX100 uses GFSK (Gaussian-filtered FSK, BT≈0.5), so real
    /// captures never have the instant, razor-sharp symbol transitions of
    /// an ideal square wave — we approximate that Gaussian smoothing here
    /// with a simple one-pole filter so this fixture is representative of
    /// what the Gardner timing loop actually has to lock onto.
    fn make_deviation_audio(
        sample_rate: u32,
        symbol_rate: f64,
        bits: &[bool],
        freq_dev: f32,
    ) -> AudioSamples {
        let sps = sample_rate as f64 / symbol_rate;
        let num_samples = (bits.len() as f64 * sps).ceil() as usize;
        let square: Vec<f32> = (0..num_samples)
            .map(|i| {
                let bit_idx = (i as f64 / sps).floor() as usize;
                if bits.get(bit_idx).copied().unwrap_or(false) {
                    freq_dev
                } else {
                    -freq_dev
                }
            })
            .collect();

        // One-pole smoothing (~symbol-rate cutoff) to emulate GFSK's
        // Gaussian pulse shaping.
        let alpha = 1.0 - (-1.0 / (sps as f32)).exp();
        let mut y = 0.0f32;
        let samples: Vec<f32> = square
            .iter()
            .map(|&x| {
                y += alpha * (x - y);
                y
            })
            .collect();

        AudioSamples {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn test_lpf_attenuates_high_freq() {
        // Inject a signal above the LPF cutoff (20 kHz) — should be attenuated
        let fs = 48_000.0f64;
        let t: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 20_000.0 * i as f32 / fs as f32).sin())
            .collect();
        let filtered = lowpass_filter(&t, fs, 14_400.0);
        let rms_in: f32 = (t.iter().map(|s| s * s).sum::<f32>() / t.len() as f32).sqrt();
        let rms_out: f32 =
            (filtered.iter().map(|s| s * s).sum::<f32>() / filtered.len() as f32).sqrt();
        // Should be significantly attenuated (>6 dB = factor 2)
        assert!(
            rms_out < rms_in / 2.0,
            "20 kHz signal should be attenuated through 14.4 kHz LPF: in={:.4}, out={:.4}",
            rms_in,
            rms_out
        );
    }

    #[test]
    fn test_full_pipeline_recovers_bits() {
        // Synthesise a pseudo-random bit pattern (not a plain alternation,
        // so a phase/count bug can't accidentally look correct) at 9600
        // baud / 48 kHz and check the pipeline recovers both the right
        // number of symbols *and* the right values.
        let pattern: Vec<bool> = (0..200u32)
            .map(|i| i.wrapping_mul(2654435761).rotate_left(13) % 5 < 2)
            .collect();
        let audio = make_deviation_audio(48_000, 9600.0, &pattern, 3200.0);
        let result = fm_discriminate_and_filter(&audio);

        // Allow ±10% symbol count variation (timing loop startup)
        let expected = pattern.len();
        let got = result.bits.len();
        let ratio = got as f64 / expected as f64;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Expected ~{} symbols, got {} (ratio {:.2})",
            expected,
            got,
            ratio
        );

        // Align the recovered bits against the known pattern (skip the
        // first few symbols, which absorb the timing loop's startup
        // transient) and check most of them match.
        let skip = 10;
        let compare_len = (result.bits.len() - skip).min(pattern.len() - skip);
        let matches = (0..compare_len)
            .filter(|&i| result.bits[skip + i] == pattern[skip + i])
            .count();
        let match_ratio = matches as f64 / compare_len as f64;
        assert!(
            match_ratio > 0.95,
            "Recovered bits should mostly match the transmitted pattern, got {:.1}% match",
            match_ratio * 100.0
        );
    }
}

