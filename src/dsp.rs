//! DSP pipeline: FM discriminator → low-pass filter → symbol timing → bit decisions.
//!
//! ## Pipeline overview
//!
//! ```text
//! f32 audio samples (real IF from SDR)
//!      │
//!      ▼
//! [FM Discriminator]   arg(s[n] * conj(s[n-1]))  →  instantaneous freq deviation
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
//! The SatNOGS audio represents the *already FM-demodulated* IF, so the
//! "FM discriminator" here is a classic complex differential detector applied
//! to the analytic signal. For real-valued audio input (one channel, no
//! quadrature) we first form the analytic signal via a Hilbert transform
//! approximated by a 64-tap FIR. This yields I+jQ from which the
//! instantaneous frequency can be extracted.

use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type};
use num_complex::Complex32;

use crate::audio::AudioSamples;

/// Output of the DSP front-end: a stream of NRZ bits and the recovered
/// symbol rate (should be ≈ 9600 Hz).
pub struct BitStream {
    /// NRZ bits, MSB first, as recovered by the symbol timing loop.
    pub bits: Vec<bool>,
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

    // 1. Build analytic signal (real → complex) via Hilbert FIR
    let analytic = hilbert_transform(&audio.samples);

    // 2. FM discriminator: instantaneous frequency
    let freq_deviation = fm_discriminator(&analytic);

    // 3. Low-pass filter at 1.5 × symbol_rate to strip out-of-band noise
    let lpf_cutoff = symbol_rate * 1.5; // 14 400 Hz
    let filtered = lowpass_filter(&freq_deviation, fs, lpf_cutoff);

    // 4. Gardner timing error detector + interpolated sampling
    let (symbols, recovered_rate) = gardner_ted(&filtered, fs, symbol_rate);

    // 5. Hard decision (slicer): positive deviation → 1, negative → 0
    let bits: Vec<bool> = symbols.iter().map(|&s| s >= 0.0).collect();

    BitStream {
        bits,
        recovered_symbol_rate: recovered_rate,
    }
}

// ---------------------------------------------------------------------------
// Step 1: Hilbert transform (real → analytic signal)
// ---------------------------------------------------------------------------
//
// A 64-tap Type-III linear-phase FIR Hilbert transformer. The analytic signal
// is: x_a[n] = x[n] + j * H{x[n]}  where H{} is the Hilbert transform.
//
// Coefficients for a 64-tap Hilbert FIR (odd indices only, even taps = 0):
// h[k] = (2/π) * sin²(πk/2) / k   for k odd, 0 for k even
// Window-weighted with a Hann window for side-lobe suppression.

const HILBERT_TAPS: usize = 64;

fn hilbert_fir_coefficients() -> [f32; HILBERT_TAPS] {
    use std::f64::consts::PI;
    let mut h = [0.0f32; HILBERT_TAPS];
    let half = (HILBERT_TAPS / 2) as i64;

    for i in 0..HILBERT_TAPS {
        let n = i as i64 - half;
        if n == 0 || n % 2 == 0 {
            continue; // zero at even taps and centre
        }
        let k = n as f64;
        // Hilbert ideal: 2/(π k) for odd k
        let ideal = 2.0 / (PI * k);
        // Hann window: 0.5 - 0.5*cos(2π(i+0.5)/N)
        let w = 0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / HILBERT_TAPS as f64).cos();
        h[i] = (ideal * w) as f32;
    }
    h
}

/// Convert a real-valued signal to its analytic (complex) representation.
/// The real part is the original signal (delayed by HILBERT_TAPS/2 samples),
/// the imaginary part is the Hilbert-transformed signal.
fn hilbert_transform(input: &[f32]) -> Vec<Complex32> {
    let h = hilbert_fir_coefficients();
    let delay = HILBERT_TAPS / 2; // group delay in samples
    let n = input.len();
    let mut out = vec![Complex32::new(0.0, 0.0); n];

    for i in 0..n {
        // Real part: delayed original signal
        let re = if i >= delay { input[i - delay] } else { 0.0 };

        // Imaginary part: FIR convolution with Hilbert kernel
        let mut im = 0.0f32;
        for (k, &coeff) in h.iter().enumerate() {
            if i >= k {
                im += coeff * input[i - k];
            }
        }
        out[i] = Complex32::new(re, im);
    }
    out
}

// ---------------------------------------------------------------------------
// Step 2: FM discriminator
// ---------------------------------------------------------------------------
//
// Classic complex differential detector:
//   f[n] = arg( x[n] · conj(x[n-1]) )  / (2π · Ts)
//
// This gives instantaneous frequency relative to the carrier. For a ±3200 Hz
// FSK signal the output swings between approximately ±3200/fs.

fn fm_discriminator(analytic: &[Complex32]) -> Vec<f32> {
    let mut out = vec![0.0f32; analytic.len()];
    for i in 1..analytic.len() {
        let product = analytic[i] * analytic[i - 1].conj();
        // arg() gives phase difference in radians; scale to [-1, 1] range
        out[i] = product.arg() / std::f32::consts::PI;
    }
    out
}

// ---------------------------------------------------------------------------
// Step 3: Biquad low-pass filter
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
// Step 4: Gardner timing error detector
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

fn gardner_ted(input: &[f32], fs: f64, symbol_rate: f64) -> (Vec<f32>, f64) {
    let sps = fs / symbol_rate; // samples per symbol (e.g. 5.0 for 48000/9600)

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

        // PI loop filter
        int_err += beta * error;
        let correction = alpha * error + int_err;

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

    (symbols, recovered_rate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioSamples;

    fn make_fsk_audio(
        sample_rate: u32,
        symbol_rate: f64,
        bits: &[bool],
        freq_dev: f32,
    ) -> AudioSamples {
        use std::f32::consts::TAU;
        let sps = sample_rate as f64 / symbol_rate;
        let num_samples = (bits.len() as f64 * sps).ceil() as usize;
        let mut samples = vec![0.0f32; num_samples];
        let mut phase = 0.0f32;
        let fs = sample_rate as f32;

        for (i, sample) in samples.iter_mut().enumerate() {
            let bit_idx = (i as f64 / sps).floor() as usize;
            let freq = if bits.get(bit_idx).copied().unwrap_or(false) {
                freq_dev
            } else {
                -freq_dev
            };
            phase += TAU * freq / fs;
            *sample = phase.sin();
        }

        AudioSamples {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn test_fm_discriminator_polarity() {
        // A constant frequency above carrier → all positive output
        let sample_rate = 48_000u32;
        let audio = make_fsk_audio(sample_rate, 9600.0, &[true; 200], 3200.0);
        let analytic = hilbert_transform(&audio.samples);
        let disc = fm_discriminator(&analytic);
        // Skip the first few samples (filter startup transients)
        let mean: f32 = disc[50..].iter().sum::<f32>() / (disc.len() - 50) as f32;
        assert!(
            mean > 0.0,
            "Positive deviation should produce positive discriminator output, got {}",
            mean
        );
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
        // Synthesise 100 alternating bits at 9600 baud / 48 kHz and check that
        // the pipeline recovers roughly the right number of symbols.
        let pattern: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let audio = make_fsk_audio(48_000, 9600.0, &pattern, 3200.0);
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
    }
}
