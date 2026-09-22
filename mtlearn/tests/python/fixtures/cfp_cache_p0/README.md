# Frozen CFP cache reference — P0

Generated before the cache refactoring on 2026-09-09 using source revision
`fd29ec8b39ff7bdb1cb30d4b89a02b9b37a2224d`.

`baseline.pt` contains five synthetic float32 images, targets, fixed training/test
indices and expected outputs, losses, gradients, statistics and public contracts
for twelve configurations. The three training images alone determine the
normalization statistics. Each case also has a real legacy checkpoint and a
separately saved statistics file. `manifest.json` records the SHA-256 digest of
all `.pt` files and comparison tolerances. These files contain synthetic data,
not images from the real dataset.

Run the tests from the repository root:

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py test -- -q \
  mtlearn/tests/python/test_cfp_cache_baseline.py
```

Historical P0 outcome on the recorded machine: **61 passed, 3 xfailed**.
P1 fixes those restoration defects; the current reference test suite now expects
**64 passed**, with the original fixture values and hashes unchanged. MPS cases
are skipped when the device is unavailable. The former xfails captured float64 statistics deserialization defects; P1
replaces them with positive restoration assertions. MPS numerical equivalence is
tested using freshly fitted training statistics and the frozen model parameters;
separate restoration tests also verify the original artifacts on MPS after P1.

Do not regenerate these files to accommodate changes in implementation results.
For auditing the producer on the original revision/build, use a new directory:

```bash
/opt/anaconda3/bin/python scripts/benchmarks/cfp_cache_p0.py freeze \
  --output build/cfp-cache-p0-audit/fixtures \
  --provenance build/cfp-cache-p0-audit/environment.json
```

Reproduction should compare tensor values/contracts; serialized file bytes may
differ across platforms or serialization versions. The checked-in manifest
protects the identity of the original artifacts.
