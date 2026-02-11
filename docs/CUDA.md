# CUDA Notes (faster-whisper / ctranslate2)

This project treats CUDA as **available** only when the installed `ctranslate2` runtime reports one or more CUDA devices.

- Probe used by `--device auto` and `--device cuda`: `ctranslate2.get_cuda_device_count()`.
- `--device auto`: selects `cuda` if count > 0, otherwise `cpu`.
- `--device cuda`: fails fast if count is 0 (or CUDA support is missing).

## Common Windows failure cases

- Installed a CPU-only `ctranslate2` wheel (no CUDA kernels included).
- NVIDIA driver is missing/outdated for your CUDA runtime.
- CUDA/cuDNN runtime DLLs are missing from `PATH`.
- Running inside a Python environment that cannot see the GPU driver/runtime.

## Quick verification checks

1. Confirm GPU + driver:

```bash
nvidia-smi
```

2. Confirm `ctranslate2` can see CUDA devices:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

3. Run locator in deterministic auto mode and inspect verbose reason:

```bash
cutscene-locator ... --asr-backend faster-whisper --device auto --verbose
```

If CUDA is unavailable, use `--device cpu` or install a CUDA-enabled `ctranslate2` build.
