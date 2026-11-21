# Fashion-MNIST DDPM

A modular PyTorch project for training an unconditional Fashion-MNIST DDPM, tracking runs and model artifacts with Weights & Biases, and serving generated images through FastAPI.

## Project structure

```text
src/ddpm/
  api/          FastAPI routes, schemas, and inference service
  training/     CLI, configuration, dataset loading, checkpoints, and training loop
  diffusion.py  Shared forward/reverse diffusion implementation
  model.py      Shared DDPM U-Net
```

The API retains `GET /health` and `POST /generate`. The generation request accepts `num_samples` (1–8) and an optional `seed`; each returned image is a base64-encoded PNG.

## Train on Kaggle

Create a Kaggle Notebook with a GPU accelerator. Attach a Fashion-MNIST dataset that contains either the CSV file `fashion-mnist_train.csv` (785 columns: label plus 784 pixels) or torchvision's FashionMNIST files. Kaggle commonly mounts attached datasets under `/kaggle/input/<dataset-folder>`.

Add a Kaggle Notebook secret named `WANDB_API_KEY` containing your W&B API key. In notebook cells:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Then clone and install the project, and train using the mounted dataset:

```bash
!git clone https://github.com/Irfan140/DDPM.git
%cd DDPM
!pip install -e .
!ddpm-train --data-dir /kaggle/input/fashion-mnist \
    --checkpoint /kaggle/working/ddpm_fashion_mnist.pth \
    --epochs 20 --batch-size 128 --wandb-project fashion-mnist-ddpm
```

The CSV loader searches recursively below `--data-dir`, so use the attached dataset's actual folder name if it differs from `fashion-mnist`. Set `WANDB_ENTITY` or pass `--wandb-entity` when logging to a W&B team. Training logs epoch loss and uploads the final checkpoint as a W&B model artifact. Checkpoints are written after each epoch, with the best training-loss checkpoint saved as `best_ddpm_fashion_mnist.pth` alongside it.

For a local run, install with `uv sync` and use `uv run ddpm-train --data-dir ./data --download`. W&B tracking is enabled by default; authenticate with `wandb login` or `WANDB_API_KEY`. Use `--no-wandb` for an offline run.

## Run inference

Point the API at the trained checkpoint. In Kaggle, download `/kaggle/working/ddpm_fashion_mnist.pth` from the notebook output, then run locally:

```powershell
$env:DDPM_WEIGHTS = "C:\path\to\ddpm_fashion_mnist.pth"
uv sync
uv run uvicorn ddpm.api.main:app --host 0.0.0.0 --port 8000
```

If no `DDPM_WEIGHTS` is set, the API looks for `ddpm_fashion_mnist.pth` at the repository root. Open `http://127.0.0.1:8000/docs` for interactive API documentation. Generation uses the timestep count saved in the checkpoint (or 1,500 for the original notebook weights).

Example request:

```json
{
  "num_samples": 2,
  "seed": 1234
}
```

The model is unconditional and cannot be prompted for a specific clothing class. Sampling runs the complete reverse diffusion schedule, so generation can take time, especially on CPU.
