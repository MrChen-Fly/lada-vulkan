# BasicVSR++ Vulkan Benchmarks

This directory contains benchmark and profiling scripts for the Vulkan
BasicVSR++ runtime. It is separate from `test/` so regression tests and
measurement tooling stay clearly separated.

Conventions:

- Generated JSON outputs and preview media are runtime artifacts, not source.
- Benchmark outputs should go to `.helloagents/tmp/` or a system temp directory.
- When measurements need to be regenerated, rerun the benchmark scripts directly.
