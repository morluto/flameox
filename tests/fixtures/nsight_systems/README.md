# Certified Nsight Systems SQLite fixture

`nsight-2025.5.2.sqlite` is an official SQLite export produced from the
`nsight_smoke.cu` workload with NVIDIA Nsight Systems
`2025.5.2.266-255236693005v0`:

```console
nsys profile --sample=none --cpuctxsw=none \
  --trace=cuda,osrt,nvtx --cuda-graph-trace=graph \
  --output=flameox-smoke nsight-smoke
nsys export --type=sqlite --force-overwrite=true \
  --output=nsight-2025.5.2.sqlite flameox-smoke.nsys-rep
```

The fixture was captured on an NVIDIA GeForce RTX 3060 with driver 595.84 and
CUDA 13.2. The export digest is:

```text
sha256:de09b9460b9f95a2b51b66d502e28cb7cf4e5dee6b748ab6c39cb6587dcb44a3
```

The report is retained as the native structured export, with environment and
host identity metadata redacted before commit. Its timestamps are observed in
the Nsight Systems export clock and are intentionally not asserted as stable
values; the adapter test asserts the stable event classes and
correlation/stream fields instead.
