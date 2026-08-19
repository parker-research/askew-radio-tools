//! DSP pipeline: matched filter → DC block → AGC → symbol timing → bit
//! decisions.
//!
//! ## Pipeline overview
//!
//! ```text
//! f32 audio samples (already FM/GFSK-demodulated frequency deviation,
//!                     as produced by the SDR/ground-station chain)
//!      │
//!      ▼
//! [Boxcar matched filter]   length = round(fs / symbol_rate)
//!      │
//!      ▼
//! [DC blocker]               4th-order cascaded-moving-average highpass
//!      │
//!      ▼
//! [RMS AGC]                  normalises to unit RMS
//!      │
//!      ▼
//! [Gardner TED + PFB interp] symbol timing recovery loop
//!      │
//!      ▼
//! [Slicer]                   threshold at 0 → Vec<bool> NRZ bits
//! ```
//!
//! This is a close port of gr-satellites' `fsk_demodulator` (real/non-IQ
//! input path, `components/demodulators/fsk_demodulator.py`), which
//! FRONTIERSAT's flowgraph uses ahead of `ax100_deframer`:
//!
//! ```text
//! sqfilter_len = int(samp_rate / baudrate)
//! taps = np.ones(sqfilter_len) / sqfilter_len
//! self.lowpass = filter.fir_filter_fff(decimation, taps)      # boxcar
//! self.dcblock = filter.dc_blocker_ff(ceil(sps * 32), True)   # DC blocker
//! self.agc = rms_agc_f(2e-2 / sps, 1)                         # RMS AGC
//! self.clock_recovery = digital.symbol_sync_ff(
//!     digital.TED_GARDNER, sps, clk_bw, damping, ted_gain,
//!     clk_limit * sps, 1, constellation_bpsk().base(), digital.IR_PFB_NO_MF)
//! ```
//!
//! (decimation is always 1 here — `sps = fs/baudrate` is ≤ 10 for every
//! sample rate this decoder supports, which is gr-satellites' own
//! threshold for decimating ahead of the matched filter, so that path
//! isn't implemented.)
//!
//! An earlier version of this module used a generic Butterworth low-pass
//! and a hand-tuned, much-lower-bandwidth Gardner loop with plain linear
//! interpolation, plus several ad-hoc mitigations (trying multiple loop
//! gains, re-locking on short anchored windows) for the loop losing lock
//! over a multi-minute capture. Benchmarking against gr_satellites'
//! reference decoder on real SatNOGS captures showed none of that closed
//! the recall gap — this version replaces it with a direct port of
//! gr-satellites' actual filter chain and loop parameters instead of
//! guessing at replacements, including its 8-tap MMSE polyphase-filterbank
//! interpolator (see `pfb_taps.rs`) and the exact PI loop gain formula its
//! `clock_tracking_loop` uses.
//!
//! SatNOGS "audio" captures for AX100 are the *already demodulated*
//! instantaneous-frequency waveform (this is what gr-satellites'
//! `ax100_deframer` itself expects as input: "a float stream of soft
//! symbols" — the demod, typically `quadrature_demod_cf`, has already run
//! upstream in the SDR chain). So there is no FM/Hilbert discriminator
//! stage here: `audio.samples` is used directly as the frequency-deviation
//! signal, positive values meaning the "mark" tone and negative values the
//! "space" tone.

use std::collections::VecDeque;

use crate::audio::AudioSamples;
use crate::pfb_taps::PFB_INTERP_TAPS;

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
// gr-satellites `fsk_demodulator` parameters (components/demodulators/
// fsk_demodulator.py), specifically FRONTIERSAT's transmitter config: no
// explicit clk_bw/clk_limit/deviation override, so these are the class
// defaults.
// ---------------------------------------------------------------------------

const SYMBOL_RATE_HZ: f64 = 9600.0;

/// `_default_clk_rel_bw` — Gardner loop's normalized natural frequency
/// (`omega_n_norm` in `clock_tracking_loop`), i.e. loop bandwidth relative
/// to the symbol rate.
const CLK_BW: f64 = 0.06;
/// `damping=1.0` in the `symbol_sync_ff` call — critically damped.
const CLK_DAMPING: f64 = 1.0;
/// "Empiric formula for TED gain of Gardner detector" per
/// `fsk_demodulator.py`'s comment.
const CLK_TED_GAIN: f64 = 1.47;
/// `_default_clk_limit` — max allowed deviation of the average clock
/// period from nominal, as a fraction of samples-per-symbol.
const CLK_LIMIT: f64 = 0.004;

/// `rms_agc_f`'s alpha numerator: `agc_constant = 2e-2 / sps` gives "a time
/// constant of 50 symbols" per the comment in `fsk_demodulator.py`.
const AGC_ALPHA_NUMERATOR: f64 = 2e-2;
/// `rms_agc_f(agc_constant, 1)` — reference amplitude of 1.0 (not the
/// `rms_agc_f` class default of 0.5).
const AGC_REFERENCE: f32 = 1.0;

/// `ceil(sps * 32)` multiplier for the DC blocker's moving-average length.
const DC_BLOCKER_LENGTH_SYMBOLS: f64 = 32.0;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full DSP front-end on a decoded audio buffer.
///
/// `samples` must be mono f32 normalised to [-1.0, 1.0] at a sample rate
/// high enough to represent 9600 baud (≥ 19200 Hz, typically 48000 Hz).
pub fn fm_discriminate_and_filter(audio: &AudioSamples) -> BitStream {
    let fs = audio.sample_rate as f64;
    let sps = fs / SYMBOL_RATE_HZ;

    // 1. Boxcar matched filter (matched to the rectangular NRZ pulse).
    let boxcar_len = (fs / SYMBOL_RATE_HZ).floor().max(1.0) as usize;
    let matched = boxcar_matched_filter(&audio.samples, boxcar_len);
    let boxcar_delay_samples = (boxcar_len as f64 - 1.0) / 2.0;

    // 2. DC blocker (cascaded-moving-average highpass).
    let dc_length = (sps * DC_BLOCKER_LENGTH_SYMBOLS).ceil().max(1.0) as usize;
    let dc_blocked = dc_blocker(&matched, dc_length);
    let dc_delay_samples = (2 * dc_length).saturating_sub(2) as f64;

    // 3. RMS AGC.
    let agc_alpha = (AGC_ALPHA_NUMERATOR / sps) as f32;
    let agced = rms_agc(&dc_blocked, agc_alpha, AGC_REFERENCE);

    // The boxcar and DC-blocker stages are both causal, linear-phase FIR-
    // style filters with a fixed, exactly-known sample delay; the PFB
    // interpolator below introduces none (see `pfb_interpolate`'s comment).
    // `gardner_ted`'s reported symbol positions are otherwise a precise,
    // direct measurement of the original audio timeline, so subtract this
    // one systematic offset back out before converting to real time.
    let total_delay_samples = boxcar_delay_samples + dc_delay_samples;

    // 4. Gardner timing error detector + PFB-interpolated sampling.
    let (symbols, sample_positions, recovered_rate) = gardner_ted(&agced, fs, SYMBOL_RATE_HZ);

    // 5. Hard decision (slicer): positive deviation → 1, negative → 0
    let bits: Vec<bool> = symbols.iter().map(|&s| s >= 0.0).collect();
    let bit_times_ms: Vec<f64> = sample_positions
        .iter()
        .map(|&p| (p - total_delay_samples) / fs * 1000.0)
        .collect();

    BitStream {
        bits,
        bit_times_ms,
        recovered_symbol_rate: recovered_rate,
    }
}

// ---------------------------------------------------------------------------
// Step 1: Boxcar matched filter (port of fsk_demodulator.py's `self.lowpass`)
// ---------------------------------------------------------------------------

/// Causal moving-average FIR of length `taps_len`, matching GNU Radio's
/// `fir_filter_fff` with `taps = ones(taps_len)/taps_len` — implicit zero
/// history before the start of `input`, same length output.
fn boxcar_matched_filter(input: &[f32], taps_len: usize) -> Vec<f32> {
    if taps_len == 0 {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len());
    let mut sum = 0.0f32;
    let mut window: VecDeque<f32> = VecDeque::with_capacity(taps_len);
    for &x in input {
        window.push_back(x);
        sum += x;
        if window.len() > taps_len {
            sum -= window.pop_front().unwrap();
        }
        out.push(sum / taps_len as f32);
    }
    out
}

// ---------------------------------------------------------------------------
// Step 2: DC blocker (literal port of gr-filter's dc_blocker_ff, long_form)
// ---------------------------------------------------------------------------

/// Port of `gr::filter::moving_averager_f`: an efficient recursive
/// D-sample moving average that also exposes the raw input delayed by
/// `D - 1` samples (`delayed_sig`).
struct MovingAverager {
    length: usize,
    out: f32,
    out_d1: f32,
    out_d2: f32,
    delay_line: VecDeque<f32>,
}

impl MovingAverager {
    fn new(length: usize) -> Self {
        MovingAverager {
            length,
            out: 0.0,
            out_d1: 0.0,
            out_d2: 0.0,
            delay_line: VecDeque::from(vec![0.0f32; length.saturating_sub(1)]),
        }
    }

    fn filter(&mut self, x: f32) -> f32 {
        self.out_d1 = self.out;
        self.delay_line.push_back(x);
        self.out = self.delay_line.pop_front().unwrap_or(0.0);

        let y = x - self.out_d1 + self.out_d2;
        self.out_d2 = y;

        y / self.length as f32
    }

    fn delayed_sig(&self) -> f32 {
        self.out
    }
}

/// Port of `dc_blocker_ff_impl::work` with `long_form=true`: a cascade of
/// four `MovingAverager`s (an efficient 4th-order cascaded-boxcar
/// approximation of an ideal DC notch) subtracted from a matching-delay
/// copy of the raw input.
fn dc_blocker(input: &[f32], length: usize) -> Vec<f32> {
    if length <= 1 {
        return input.to_vec();
    }
    let mut ma0 = MovingAverager::new(length);
    let mut ma1 = MovingAverager::new(length);
    let mut ma2 = MovingAverager::new(length);
    let mut ma3 = MovingAverager::new(length);
    let mut delay_line: VecDeque<f32> = VecDeque::from(vec![0.0f32; length.saturating_sub(1)]);

    let mut out = Vec::with_capacity(input.len());
    for &x in input {
        let y1 = ma0.filter(x);
        let y2 = ma1.filter(y1);
        let y3 = ma2.filter(y2);
        let y4 = ma3.filter(y3);

        delay_line.push_back(ma0.delayed_sig());
        let d = delay_line.pop_front().unwrap_or(0.0);

        out.push(d - y4);
    }
    out
}

// ---------------------------------------------------------------------------
// Step 3: RMS AGC (port of gr-satellites' hier/rms_agc_f.py)
// ---------------------------------------------------------------------------

/// Port of `rms_agc_f`: `blocks.rms_ff(alpha)` (single-pole running RMS)
/// feeding `output = input / (rms / reference + 1e-19)`.
fn rms_agc(input: &[f32], alpha: f32, reference: f32) -> Vec<f32> {
    let beta = 1.0 - alpha;
    let mut avg = 0.0f32;
    input
        .iter()
        .map(|&x| {
            avg = beta * avg + alpha * x * x;
            let rms = avg.sqrt();
            x / (rms / reference + 1e-19)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Step 4: 8-tap MMSE polyphase-filterbank interpolator (port of GNU Radio's
// `interp_resampler_pfb_no_mf_ff`, `digital.IR_PFB_NO_MF`)
// ---------------------------------------------------------------------------

/// Number of polyphase arms (`n_filters`, `symbol_sync_ff`'s default of
/// 128, already a power of 2 so it's used as-is — see
/// `interp_resampler_pfb_no_mf_ff`'s constructor, which rounds up to the
/// next power of 2 and caps at `NSTEPS`, both no-ops at 128).
const PFB_N_FILTERS: usize = 128;

/// Interpolate `input`'s value at fractional position `pos` using the
/// 8-tap MMSE polyphase filter bank (see `pfb_taps.rs`). Positions outside
/// `input`'s bounds read as zero (matching a zero-padded/zero-history
/// signal, same convention as [`boxcar_matched_filter`]).
///
/// This introduces no net delay: tap column 4 (of 0..7, offsets -4..+3)
/// is `input[floor(pos)]` at `mu = 0` (row 0 of the table is a unit
/// impulse there), so the interpolated value at `pos` is referenced
/// directly against the same timeline `pos` is expressed in.
fn pfb_interpolate(input: &[f32], pos: f64) -> f32 {
    let base = pos.floor();
    let mu = (pos - base) as f32;
    let base = base as i64;

    let arm = (mu * PFB_N_FILTERS as f32).round() as usize;
    let taps = &PFB_INTERP_TAPS[arm.min(PFB_INTERP_TAPS.len() - 1)];

    let mut acc = 0.0f32;
    for (k, &tap) in taps.iter().enumerate() {
        let idx = base + k as i64 - 4;
        if idx >= 0 && (idx as usize) < input.len() {
            acc += tap * input[idx as usize];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Step 4: Gardner timing error detector (port of GNU Radio's
// `ted_gardner` + `clock_tracking_loop`, driven as `symbol_sync_ff` does
// with `TED_GARDNER` — inputs_per_symbol=2, i.e. one mid-symbol and one
// on-symbol interpolated sample per symbol period)
// ---------------------------------------------------------------------------
//
// The Gardner TED is a decision-directed timing error detector that works
// on passband signals without a separate carrier reference. It estimates
// the timing error from mid-symbol and on-symbol samples:
//
//   e[k] = (y[k - T] - y[k]) · y[k - T/2]
//
// where y[k] is the current on-symbol sample, y[k - T] the previous
// on-symbol sample, and y[k - T/2] the mid-symbol sample between them —
// this is `ted_gardner::compute_error_ff`'s `(d_input[2] - d_input[0]) *
// d_input[1]` with `d_input[2]` the older on-symbol sample.
//
// The loop filter is `clock_tracking_loop`'s PI controller, with alpha
// (proportional) and beta (integral) gains derived from the loop
// bandwidth, damping factor and TED gain by `update_gains`'s exact
// formula (ported in `pi_loop_gains` below) rather than hand-tuned.

/// Port of `clock_tracking_loop::update_gains` — maps (loop bandwidth,
/// damping factor, TED gain) to the PI filter's (alpha, beta) gains via
/// the standard 2nd-order digital control loop design equations.
fn pi_loop_gains(damping: f64, loop_bw: f64, ted_gain: f64) -> (f64, f64) {
    let zeta = damping;
    let omega_n_t = loop_bw;
    let zeta_omega_n_t = zeta * omega_n_t;

    let k0 = 2.0 / ted_gain;
    let k1 = (-zeta_omega_n_t).exp();
    let sinh_zeta_omega_n_t = zeta_omega_n_t.sinh();

    let cosx_omega_d_t = match zeta.partial_cmp(&1.0).unwrap() {
        std::cmp::Ordering::Greater => {
            let omega_d_t = omega_n_t * (zeta * zeta - 1.0).sqrt();
            omega_d_t.cosh()
        }
        std::cmp::Ordering::Equal => 1.0,
        std::cmp::Ordering::Less => {
            let omega_d_t = omega_n_t * (1.0 - zeta * zeta).sqrt();
            omega_d_t.cos()
        }
    };

    let alpha = k0 * k1 * sinh_zeta_omega_n_t;
    let beta = k0 * (1.0 - k1 * (sinh_zeta_omega_n_t + cosx_omega_d_t));
    (alpha, beta)
}

fn gardner_ted(input: &[f32], fs: f64, symbol_rate: f64) -> (Vec<f32>, Vec<f64>, f64) {
    let sps = fs / symbol_rate;
    let max_deviation = CLK_LIMIT * sps;
    let (alpha, beta) = pi_loop_gains(CLK_DAMPING, CLK_BW, CLK_TED_GAIN);

    let mut avg_period = sps;
    let mut inst_period = sps;
    let mut idx = sps; // current on-symbol strobe position

    let mut symbols: Vec<f32> = Vec::with_capacity(input.len() / sps as usize);
    let mut sample_positions: Vec<f64> = Vec::new();

    while idx + sps < input.len() as f64 {
        let y_on = pfb_interpolate(input, idx);
        let y_mid_prev = pfb_interpolate(input, idx - inst_period / 2.0);
        let y_prev = pfb_interpolate(input, idx - inst_period);

        // Gardner TED error: uses the *previous* on-symbol sample to avoid
        // decision feedback.
        let error = ((y_prev - y_on) * y_mid_prev) as f64;

        // PI loop filter (`clock_tracking_loop::advance_loop`).
        avg_period = (avg_period + beta * error).clamp(sps - max_deviation, sps + max_deviation);
        inst_period = avg_period + alpha * error;
        if inst_period <= 0.0 {
            inst_period = avg_period;
        }

        symbols.push(y_on);
        sample_positions.push(idx);

        idx += inst_period;
    }

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
    fn test_pi_loop_gains_match_fsk_demodulator_defaults() {
        // Cross-check against an independent Python re-implementation of
        // clock_tracking_loop::update_gains for gr-satellites'
        // fsk_demodulator.py defaults (damping=1.0, clk_bw=0.06,
        // ted_gain=1.47): alpha ≈ 0.07692487, beta ≈ 0.00230705.
        let (alpha, beta) = pi_loop_gains(1.0, 0.06, 1.47);
        assert!(
            (alpha - 0.07692487).abs() < 1e-6,
            "alpha = {alpha}, expected ~0.07692487"
        );
        assert!(
            (beta - 0.00230705).abs() < 1e-6,
            "beta = {beta}, expected ~0.00230705"
        );
    }

    #[test]
    fn test_pfb_interpolate_is_identity_at_integer_positions() {
        let input = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for i in 4..8 {
            let got = pfb_interpolate(&input, i as f64);
            assert!(
                (got - input[i]).abs() < 1e-4,
                "pfb_interpolate at integer position {i} should be ~identity, got {got}, expected {}",
                input[i]
            );
        }
    }

    #[test]
    fn test_boxcar_matched_filter_averages() {
        let input = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let out = boxcar_matched_filter(&input, 5);
        // Zero-padded history means only the last sample sees the full window.
        assert!((out[4] - 1.0).abs() < 1e-6);
        assert!((out[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_rms_agc_normalises_amplitude() {
        let input = vec![2.0f32; 2000];
        let out = rms_agc(&input, 0.01, 1.0);
        // After convergence, output amplitude should approach the reference (1.0).
        let tail_avg: f32 = out[1000..].iter().sum::<f32>() / 1000.0;
        assert!(
            (tail_avg - 1.0).abs() < 0.05,
            "AGC should converge output amplitude to ~1.0, got {tail_avg}"
        );
    }

    #[test]
    fn test_full_pipeline_recovers_bits() {
        // Synthesise a pseudo-random bit pattern (not a plain alternation,
        // so a phase/count bug can't accidentally look correct) at 9600
        // baud / 48 kHz and check the pipeline recovers both the right
        // number of symbols *and* the right values.
        //
        // Long enough that the DC blocker's settling transient (4
        // cascaded ~160-sample moving averages, per `DC_BLOCKER_LENGTH_
        // SYMBOLS`) is a small fraction of the signal — real captures are
        // minutes long, so this only matters for a short synthetic test.
        let pattern: Vec<bool> = (0..1500u32)
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

        // Find the best constant index offset aligning the recovered
        // bits against the pattern. The synthetic signal here has no
        // noise pre-roll before the pattern starts (unlike a real
        // capture, which has minutes of it), so the loop's first strobes
        // land in the matched-filter/DC-blocker's own settling region —
        // a fixed number of "virtual" symbols before the pattern's first
        // real bit — hence a constant offset rather than a 1:1 index
        // correspondence.
        let mut best_ratio = 0.0f64;
        for shift in -300i64..300 {
            let mut matches = 0;
            let mut compared = 0;
            for (i, &bit) in result.bits.iter().enumerate() {
                let j = i as i64 + shift;
                if j < 0 || j as usize >= pattern.len() {
                    continue;
                }
                compared += 1;
                if bit == pattern[j as usize] {
                    matches += 1;
                }
            }
            if compared > pattern.len() / 2 {
                best_ratio = best_ratio.max(matches as f64 / compared as f64);
            }
        }
        assert!(
            best_ratio > 0.95,
            "Recovered bits should mostly match the transmitted pattern at some \
             constant offset, best match was only {:.1}%",
            best_ratio * 100.0
        );
    }
}
