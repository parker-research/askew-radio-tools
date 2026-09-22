# askew-radio-tools

[![crates.io](https://img.shields.io/crates/v/askew-radio-tools.svg)](https://crates.io/crates/askew-radio-tools)
[![docs.rs](https://img.shields.io/docsrs/askew-radio-tools)](https://docs.rs/askew-radio-tools)
[![CI](https://github.com/parker-research/askew-radio-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/parker-research/askew-radio-tools/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/askew-radio-tools.svg)](LICENSE)

DSP demodulator for AX100 Mode 5 radio packets, and more, written in Rust.

## Tools

1. `askew_demod_from_file`: Decodes SatNOGS audio captures into JSONL frames.

## Future Directions

Please open an Issue if you're interested in any of the following features:

1. Implement uplink/modulation of data to packets
2. Implement more encoding schemes from `gr_satellites` and similar
3. Add support for receiving data live from a Software Defined Radio (SDR)
4. Emit data via TCP or UDP
5. Anything else?

## History, Context

This project spawned out of desire for the best SatNOGS data decoder during
mission operations for the [CalgaryToSpace FrontierSat](https://github.com/CalgaryToSpace/CTS-SAT-1-Wiki/wiki)
mission.

On the scale of "quality code" to "quality result", this project likely lies
a little closer to focusing on the quality of the result, and is backed strongly
by regression tests. It is heavily developed by the use of AI, for better or
for worse.
