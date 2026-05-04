//! Audio file loading for WAV and OGG Vorbis files from SatNOGS.
//!
//! Uses `symphonia` as the unified decoder backend. Both formats share
//! identical handling — the codec is auto-detected from the file header.
//!
//! SatNOGS typically produces:
//!   - 48000 Hz mono WAV (from GNU Radio flowgraphs)
//!   - 44100 Hz or 48000 Hz mono OGG Vorbis (from some station configs)
//!
//! Samples are returned as f32 in the range [-1.0, 1.0]. For a real
//! SDR baseband IF capture the input is already real-valued audio
//! representing the FM-demodulated IF, so mono is expected. If stereo
//! is found, only the left channel is used.

use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::DecodeError;

/// Decoded audio samples and the sample rate they were captured at.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Interleaved f32 samples, normalised to [-1.0, 1.0].
    /// For mono audio this is just a flat Vec. For stereo only
    /// the left channel has been kept.
    pub samples: Vec<f32>,

    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,

    /// Number of channels in the *original* file (informational).
    pub channels: usize,
}

impl AudioSamples {
    /// Samples-per-symbol at 9600 baud.
    pub fn samples_per_symbol(&self) -> f64 {
        self.sample_rate as f64 / 9600.0
    }
}

/// Load a WAV or OGG Vorbis audio file and return normalised f32 samples.
///
/// Auto-detects the format from the file extension and magic bytes.
/// Stereo files are downmixed to mono by discarding the right channel.
///
/// # Errors
/// Returns [`DecodeError::Audio`] on any I/O or codec failure,
/// [`DecodeError::UnsupportedFormat`] if the file is neither WAV nor OGG,
/// and [`DecodeError::SampleRateTooLow`] if the sample rate cannot
/// represent 9600 baud symbols (must be ≥ 19200 Hz for ≥2 samples/symbol).
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<AudioSamples, DecodeError> {
    let path = path.as_ref();

    // --- Open the file and wrap in a MediaSourceStream ---
    let file = std::fs::File::open(path)
        .map_err(|e| DecodeError::Audio(format!("Cannot open {:?}: {}", path, e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Give symphonia a hint from the file extension so it can skip
    // probing formats it doesn't need to try.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    // Probe the media source — finds the demuxer (WAV or OGG container).
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| match e {
            SymphoniaError::Unsupported(s) => DecodeError::UnsupportedFormat(s.to_string()),
            other => DecodeError::Audio(other.to_string()),
        })?;

    let mut format = probed.format;

    // Find the first audio track in the container.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::Audio("No audio track found".into()))?
        .clone();

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| DecodeError::Audio("Sample rate unknown".into()))?;

    // Minimum: 2 samples per symbol at 9600 baud.
    const MIN_SAMPLE_RATE: u32 = 19_200;
    if sample_rate < MIN_SAMPLE_RATE {
        return Err(DecodeError::SampleRateTooLow {
            rate: sample_rate,
            min: MIN_SAMPLE_RATE,
        });
    }

    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let track_id = track.id;

    // Build the audio decoder for this track's codec (PCM or Vorbis).
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| DecodeError::UnsupportedFormat(e.to_string()))?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    // --- Decode loop ---
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(DecodeError::Audio(e.to_string())),
        };

        // Skip packets that belong to other tracks (cover art, etc.)
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Non-fatal: skip corrupted packets and keep going.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(DecodeError::Audio(e.to_string())),
        };

        // Initialise the sample buffer on the first decoded packet
        // (we now know the actual spec).
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;

        let buf = sample_buf.get_or_insert_with(|| SampleBuffer::new(duration, spec));
        buf.copy_interleaved_ref(decoded);

        let raw = buf.samples();

        if channels == 1 {
            // Mono: copy directly.
            all_samples.extend_from_slice(raw);
        } else {
            // Multi-channel: keep only the left (index 0) channel.
            all_samples.extend(raw.chunks(channels).map(|ch| ch[0]));
        }
    }

    if all_samples.is_empty() {
        return Err(DecodeError::Audio("No samples decoded from file".into()));
    }

    Ok(AudioSamples {
        samples: all_samples,
        sample_rate,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal valid 48 kHz mono WAV file with a sine wave and
    /// round-trip it through the loader to confirm basic plumbing works.
    #[test]
    fn test_load_mono_wav_roundtrip() {
        use std::f32::consts::TAU;

        let sample_rate: u32 = 48_000;
        let num_samples: usize = 48_000; // 1 second
        let freq = 1000.0_f32; // 1 kHz sine

        // Build raw PCM i16 samples
        let pcm: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (f32::sin(TAU * freq * t) * i16::MAX as f32) as i16
            })
            .collect();

        // Write a minimal WAV file to a temp path
        let path = "/tmp/test_beacon_audio.wav";
        write_minimal_wav(path, sample_rate, &pcm).expect("write test WAV");

        let audio = load_audio(path).expect("load test WAV");
        assert_eq!(audio.sample_rate, sample_rate);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), num_samples);

        // Samples should be normalised near [-1, 1]
        let max = audio
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max > 0.9 && max <= 1.0, "max sample out of range: {}", max);
    }

    #[test]
    fn test_sample_rate_too_low_rejected() {
        // Write a 8000 Hz WAV — below the 19200 Hz minimum
        let sample_rate: u32 = 8_000;
        let pcm: Vec<i16> = vec![0i16; 8000];
        let path = "/tmp/test_low_rate.wav";
        write_minimal_wav(path, sample_rate, &pcm).expect("write low-rate WAV");

        let result = load_audio(path);
        assert!(matches!(result, Err(DecodeError::SampleRateTooLow { .. })));
    }

    /// Write a bare-minimum WAV (PCM, mono, i16) — no external crates needed.
    fn write_minimal_wav(path: &str, sample_rate: u32, samples: &[i16]) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        let num_samples = samples.len() as u32;
        let data_size = num_samples * 2; // 2 bytes per i16
        let chunk_size = 36 + data_size;

        // RIFF header
        f.write_all(b"RIFF")?;
        f.write_all(&chunk_size.to_le_bytes())?;
        f.write_all(b"WAVE")?;

        // fmt  sub-chunk
        f.write_all(b"fmt ")?;
        f.write_all(&16u32.to_le_bytes())?; // sub-chunk size
        f.write_all(&1u16.to_le_bytes())?; // PCM
        f.write_all(&1u16.to_le_bytes())?; // mono
        f.write_all(&sample_rate.to_le_bytes())?;
        f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
        f.write_all(&2u16.to_le_bytes())?; // block align
        f.write_all(&16u16.to_le_bytes())?; // bits per sample

        // data sub-chunk
        f.write_all(b"data")?;
        f.write_all(&data_size.to_le_bytes())?;
        for s in samples {
            f.write_all(&s.to_le_bytes())?;
        }
        Ok(())
    }
}
