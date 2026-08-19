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
/// to the symbol rate. This is a single, fixed, continuous loop running
/// over a whole multi-minute capture — even well-tuned (this ports
/// gr-satellites' exact defaults, not a guess), it can still momentarily
/// lose lock at some specific point in an 11-minute file and miss a real
/// burst there. Benchmarking against gr_satellites' reference decoder on
/// real captures confirmed this happens rarely but for real: one burst
/// undecoded at the default bandwidth decoded cleanly at every one of
/// several other bandwidths tried. So [`CLK_BW_CANDIDATES`] runs a small,
/// cheap ensemble of bandwidths and merges whatever each recovers
/// (deduplicated downstream by payload bytes), rather than relying on a
/// single value.
const CLK_BW: f64 = CLK_BW_CANDIDATES[0];

/// Loop bandwidths to try (see [`CLK_BW`]'s comment). The first is
/// gr-satellites' actual default (`_default_clk_rel_bw`); the rest are
/// arbitrary but spread around it — any one of them recovering a frame
/// the default misses is enough, so precisely which extra values are
/// used matters much less than having a couple of options.
pub const CLK_BW_CANDIDATES: &[f64] = &[0.06, 0.03, 0.12];

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

/// Run the full DSP front-end on a decoded audio buffer using the default
/// (gr-satellites-matching) loop bandwidth. See [`CLK_BW_CANDIDATES`] for
/// running with an alternate bandwidth.
///
/// `samples` must be mono f32 normalised to [-1.0, 1.0] at a sample rate
/// high enough to represent 9600 baud (≥ 19200 Hz, typically 48000 Hz).
pub fn fm_discriminate_and_filter(audio: &AudioSamples) -> BitStream {
    fm_discriminate_and_filter_with_bw(audio, CLK_BW)
}

/// Same as [`fm_discriminate_and_filter`], but with an explicit Gardner
/// loop bandwidth (see [`CLK_BW_CANDIDATES`]).
pub fn fm_discriminate_and_filter_with_bw(audio: &AudioSamples, clk_bw: f64) -> BitStream {
    bitstream_from_front_end(&FrontEnd::compute(audio), clk_bw)
}

/// Run [`fm_discriminate_and_filter_with_bw`] for every bandwidth in `bws`,
/// sharing one front-end computation (steps 1-3 below) across all of them
/// instead of repeating it per bandwidth — those steps don't depend on
/// `clk_bw` at all, only step 4 does. This is the same output as calling
/// `fm_discriminate_and_filter_with_bw` once per entry in `bws`, just
/// without redoing the shared ~2/3 of the work each time.
pub fn fm_discriminate_and_filter_multi_bw(audio: &AudioSamples, bws: &[f64]) -> Vec<BitStream> {
    let front_end = FrontEnd::compute(audio);
    bws.iter()
        .map(|&clk_bw| bitstream_from_front_end(&front_end, clk_bw))
        .collect()
}

/// Output of steps 1-3 (boxcar matched filter, DC blocker, RMS AGC) — the
/// part of the pipeline that's the same regardless of the Gardner loop
/// bandwidth used in step 4.
struct FrontEnd {
    agced: Vec<f32>,
    fs: f64,
    /// See the comment in [`bitstream_from_front_end`] on why this is
    /// tracked and subtracted back out.
    total_delay_samples: f64,
}

impl FrontEnd {
    fn compute(audio: &AudioSamples) -> FrontEnd {
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

        FrontEnd {
            agced,
            fs,
            total_delay_samples: boxcar_delay_samples + dc_delay_samples,
        }
    }
}

fn bitstream_from_front_end(front_end: &FrontEnd, clk_bw: f64) -> BitStream {
    // 4. Gardner timing error detector + PFB-interpolated sampling.
    let (symbols, sample_positions, recovered_rate) =
        gardner_ted(&front_end.agced, front_end.fs, SYMBOL_RATE_HZ, clk_bw);

    // 5. Hard decision (slicer): positive deviation → 1, negative → 0
    let bits: Vec<bool> = symbols.iter().map(|&s| s >= 0.0).collect();
    // The boxcar and DC-blocker stages are both causal, linear-phase FIR-
    // style filters with a fixed, exactly-known sample delay; the PFB
    // interpolator introduces none (see `pfb_interpolate`'s comment).
    // `gardner_ted`'s reported symbol positions are otherwise a precise,
    // direct measurement of the original audio timeline, so subtract this
    // one systematic offset back out before converting to real time.
    let bit_times_ms: Vec<f64> = sample_positions
        .iter()
        .map(|&p| (p - front_end.total_delay_samples) / front_end.fs * 1000.0)
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

fn gardner_ted(input: &[f32], fs: f64, symbol_rate: f64, clk_bw: f64) -> (Vec<f32>, Vec<f64>, f64) {
    let sps = fs / symbol_rate;
    let max_deviation = CLK_LIMIT * sps;
    let (alpha, beta) = pi_loop_gains(CLK_DAMPING, clk_bw, CLK_TED_GAIN);

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
// Alternate front-end: Mueller-Müller decision-directed timing recovery
// ---------------------------------------------------------------------------
//
// A structurally different demod chain, ported from `simple_sat_ops`'
// `modem_pcm16_to_bits` (`src/dsp/modem.c`) — an independently-developed
// sister decoder for this same AX100/FrontierSat downlink. Benchmarking it
// (via its `rx_replay --forensics-report` CLI) against this decoder on the
// same real captures found it recovers real frames the chain above misses
// *entirely*: on one file, 5 of 6 repeat transmissions of a short "Could
// not set config var" message never produced even a syncword hit for us,
// at any of the [`CLK_BW_CANDIDATES`] bandwidths — not a marginal RS
// failure, a total miss.
//
// The chain differs in several structural ways, not just tuning:
//   - DC-block runs *before* the matched filter (the chain above runs it
//     after).
//   - AGC is a single static whole-file RMS, not a running/adaptive one.
//   - Timing recovery is Mueller-Müller (decision-directed: error =
//     sign(y[k-1])·y[k] - sign(y[k])·y[k-1]) with a pure proportional loop,
//     not Gardner with a 2nd-order PI loop.
//   - The strobe sample is linearly interpolated, not PFB-interpolated.
//
// None of these individually looks obviously "better" than the GNU Radio-
// matching chain above — this is a genuinely different algorithm, with a
// different error S-curve and different susceptibility to whatever made
// those particular bursts hard to track — which is exactly why running it
// as an additional ensemble member (rather than guessing which specific
// piece of the difference mattered) recovers frames the primary chain
// can't reach no matter how its parameters are tuned. See
// [`crate::pipeline::decode_audio`] for how it's merged in.

/// `modem.c`'s DC-block filter coefficient (`alpha = 0.995f`, "≈ 40 Hz
/// -3dB" at 48 kHz).
const MM_DC_BLOCK_ALPHA: f32 = 0.995;
/// `modem.c`'s Mueller-Müller loop proportional gain (`Kp = 0.10`).
const MM_LOOP_KP: f64 = 0.10;
/// `modem.c`'s per-step clamp, as a fraction of `sps` (`max_step =
/// sps_d * 0.25`).
const MM_MAX_STEP_FRACTION: f64 = 0.25;

/// Run the alternate Mueller-Müller front-end described above. Meant to be
/// merged with (not replace) [`fm_discriminate_and_filter`]'s output — see
/// the module comment above.
pub fn fm_discriminate_and_filter_mueller_muller(audio: &AudioSamples) -> BitStream {
    let fs = audio.sample_rate as f64;
    let sps = fs / SYMBOL_RATE_HZ;

    // 1. DC-block (1-pole HPF) — ahead of the matched filter here, unlike
    // the primary chain.
    let dc_blocked = one_pole_dc_block(&audio.samples, MM_DC_BLOCK_ALPHA);
    let dc_delay_samples = one_pole_dc_block_group_delay_samples(fs);

    // 2. Static whole-file RMS AGC (not adaptive/running).
    let rms = {
        let sum_sq: f64 = dc_blocked.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (sum_sq / dc_blocked.len().max(1) as f64)
            .sqrt()
            .max(1.0 / i16::MAX as f64)
    };
    let agc_inv = (1.0 / rms) as f32;

    // 3. Boxcar matched filter (length sps), AGC-scaled.
    let boxcar_len = sps.floor().max(1.0) as usize;
    let mf: Vec<f32> = boxcar_matched_filter(&dc_blocked, boxcar_len)
        .iter()
        .map(|&x| x * agc_inv)
        .collect();
    let boxcar_delay_samples = (boxcar_len as f64 - 1.0) / 2.0;

    let total_delay_samples = dc_delay_samples + boxcar_delay_samples;

    // 4. Mueller-Müller timing recovery + linearly-interpolated strobe.
    let (symbols, sample_positions) = mueller_muller_ted(&mf, sps);

    let bits: Vec<bool> = symbols.iter().map(|&s| s >= 0.0).collect();
    let bit_times_ms: Vec<f64> = sample_positions
        .iter()
        .map(|&p| (p - total_delay_samples) / fs * 1000.0)
        .collect();

    let recovered_rate = if sample_positions.len() > 1 {
        let mean_sps = (sample_positions.last().unwrap() - sample_positions[0])
            / (sample_positions.len() - 1) as f64;
        fs / mean_sps
    } else {
        SYMBOL_RATE_HZ
    };

    BitStream {
        bits,
        bit_times_ms,
        recovered_symbol_rate: recovered_rate,
    }
}

/// Port of `modem.c`'s DC-block: `y[n] = x[n] - x[n-1] + alpha*y[n-1]`.
fn one_pole_dc_block(input: &[f32], alpha: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    let mut prev_x = 0.0f32;
    let mut prev_y = 0.0f32;
    for &x in input {
        let y = x - prev_x + alpha * prev_y;
        out.push(y);
        prev_x = x;
        prev_y = y;
    }
    out
}

/// Group delay (in samples) of [`one_pole_dc_block`]'s `H(z) = (1 - z^-1) /
/// (1 - alpha*z^-1)`, evaluated (via the same numerical-derivative-of-
/// phase approach as `dsp.rs`'s earlier Butterworth analysis used) at half
/// the symbol rate — a representative frequency for where this filter's
/// *passband* behavior matters, since (unlike a low-pass) the signal band
/// here sits well above this high-pass's own cutoff (~40 Hz), not below
/// it.
fn one_pole_dc_block_group_delay_samples(fs: f64) -> f64 {
    let alpha = MM_DC_BLOCK_ALPHA as f64;
    let phase_at = |omega: f64| -> f64 {
        let (s, c) = omega.sin_cos();
        let num_phase = s.atan2(1.0 - c);
        let den_phase = (alpha * s).atan2(1.0 - alpha * c);
        num_phase - den_phase
    };
    let omega0 = 2.0 * std::f64::consts::PI * (SYMBOL_RATE_HZ / 2.0) / fs;
    let h = 1e-4_f64;
    -(phase_at(omega0 + h) - phase_at(omega0 - h)) / (2.0 * h)
}

/// Port of `modem.c`'s Mueller-Müller timing loop: a decision-directed TED
/// (`error = sign(y[k-1])·y[k] - sign(y[k])·y[k-1]`) with a pure
/// proportional loop filter (no integral term) and linear-interpolated
/// strobe sampling.
fn mueller_muller_ted(mf: &[f32], sps: f64) -> (Vec<f32>, Vec<f64>) {
    let max_step = sps * MM_MAX_STEP_FRACTION;

    let mut symbols: Vec<f32> = Vec::with_capacity((mf.len() as f64 / sps) as usize);
    let mut sample_positions: Vec<f64> = Vec::new();

    let mut pos = sps;
    let mut prev_y = 0.0f64;
    let mut prev_dec = 0.0f64;
    let mut have_prev = false;

    while pos + 1.0 < mf.len() as f64 {
        let i = pos.floor() as usize;
        let frac = pos - i as f64;
        if i + 1 >= mf.len() {
            break;
        }
        let y = mf[i] as f64 * (1.0 - frac) + mf[i + 1] as f64 * frac;
        let dec = if y >= 0.0 { 1.0 } else { -1.0 };

        symbols.push(y as f32);
        sample_positions.push(pos);

        let mut advance = sps;
        if have_prev {
            let ted = prev_dec * y - dec * prev_y;
            let adj = (MM_LOOP_KP * ted).clamp(-max_step, max_step);
            advance += adj;
        }
        pos += advance;
        prev_y = y;
        prev_dec = dec;
        have_prev = true;
    }

    (symbols, sample_positions)
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
