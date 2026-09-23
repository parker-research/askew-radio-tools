default: all

fmt:
    cargo fmt --all -- --check

check:
    cargo check --all-targets

clippy:
    cargo clippy --all-targets -- -D warnings

build:
    cargo build --all-targets
    cargo build --all-targets --release

test:
    cargo test --all-targets --release

all: fmt check clippy build test
