# Fashion-MNIST DDPM inference API

This modular FastAPI service loads `ddpm_fashion_mnist.pth` and generates 28×28 grayscale images with the same unconditional DDPM U-Net and 1,500-step noise schedule used by the training notebook.

## Run

From the repository root, install the project dependencies and start the API:

```bash
uv sync
uv run uvicorn src.main:app --reload
```

The checkpoint is loaded at startup. To use another checkpoint, set `DDPM_WEIGHTS` to its path. Set `DDPM_DEVICE` to `cpu`, `cuda`, or `auto` (the default).

```powershell
$env:DDPM_WEIGHTS = "C:\path\to\checkpoint.pth"
$env:DDPM_DEVICE = "auto"
uv run uvicorn src.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`; health status is at `/health`.

## Generate images

Send `POST /generate` with an optional sample count (1–8) and seed:

```json
{
  "num_samples": 2,
  "seed": 1234
}
```

Each result contains a base64-encoded PNG in `images[].data`. Decode it as a PNG to save or display it. The response includes the seed used, including when the server chose it. Supplying the same seed reproduces the same output on the same device and software setup.

This is an unconditional model: requests cannot specify a Fashion-MNIST class. Sampling runs all 1,500 reverse diffusion steps, so generation can take some time, especially on CPU.

## Modules

- `src/model.py`: training-compatible U-Net architecture
- `src/sampling.py`: reverse DDPM sampler
- `src/service.py`: checkpoint loading and PNG encoding
- `src/schemas.py`: request and response models
- `src/main.py`: FastAPI app and routes
