# Repository Agent Rules

## Test Execution Hygiene (Windows Sandbox)

- When running any tests (`python -m unittest`, `pytest`, or similar), prefer elevated execution to avoid sandbox temp-file residue in the repository root.
- Do not run tests in a way that leaves random 8-character probe files in the project directory.

## Mandatory Post-Test Cleanup

- After tests, clean temporary probe files in:
  - repository root
  - `tmp/` directory
- Target files matching all conditions:
  - filename regex: `^[A-Za-z0-9_]{8}$`
  - very small probe size (typically 1-4 bytes)
- Also remove known probe artifacts if present:
  - `tmp/probe.txt`
  - `tmp/zz_test.tmp`
  - `tmp/x.txt`

## If Elevation Is Not Available

- Explicitly warn the user before running tests that residue files may be produced.
- Request elevation first whenever possible.
