#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)
)]
//! Binary entrypoint for the kb knowledge base CLI.

fn main() {
    std::process::exit(kb::cli_main::main_exit_code());
}
