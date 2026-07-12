# Benchmark Results Submission

Thanks for running the clean suite! The result JSONs already contain your
hardware slug, power source, and thermal notes — please answer only what
they can't capture.

## Machine

- **Device** (e.g. MacBook Pro 14" / Mac mini / iMac):
- **Chip** (auto-detected slug from the suite output, e.g. `Apple_M3_Pro_18GB_11CPU_14GPU_18ANE`):
- **Cooling**: active (has fan) / passive (MacBook Air)
- **macOS version** (`sw_vers -productVersion`):

## Run conditions

- [ ] Ran `./benchmarks/run_clean_suite.sh` unmodified from the `benchmarks` branch
- [ ] On AC power for the entire run
- [ ] No other apps in use during the run
- [ ] Machine was idle ≥30 min (or freshly booted) before starting
- Anything unusual during the run (interruptions, warnings, failed phases — paste console output if any):

## Files

- [ ] Only files under `benchmarks/results/<my-slug>/` are added — no hand-edited JSONs, no code changes in the same PR

## Optional

- Suite total runtime:
- Anything that surprised you in the numbers:
