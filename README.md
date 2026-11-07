# OcuFlow

Eyeglass reflection removal with batch and single-image inference.

## Setup

Use Python 3.10+ and PyTorch 2.4+ with a CUDA build compatible with your GPU. The batch example below uses BF16 and requires a GPU with BF16 support.

```bash
cd /home/liutao/video/RR/OcuFlow
python -m pip install 'torch>=2.4'
python -m pip install -r requirements.txt
```

## Example Files

```text
OcuFlow/
├── infer.py
├── weights/checkpoint_best.pt
├── assets/face_landmarker.task
└── examples/
    ├── test_mini/
    │   ├── pub_000654/
    │   │   ├── glare.png       # Input
    │   │   └── clean.png       # Reference; not required for inference
    │   └── ...
    ├── output/                # Batch predictions
    └── predict/               # Single-image prediction
```

Run the following commands from the project directory. The checkpoint and face detection asset must be present at the paths above.

## Batch Inference

```bash
python infer.py \
  --input examples/test_mini \
  --checkpoint weights/checkpoint_best.pt \
  --output examples/output \
  --batch_size 8 \
  --preprocess_workers 4 \
  --prefetch_batches 3 \
  --save_workers 4 \
  --patch_backend gpu \
  --amp bf16 \
  --grid_solver fast
```

Finds `glare.png` recursively and saves each prediction as `examples/output/<sample>/pred.png`.

## Single-Image Inference

```bash
python infer.py \
  --input examples/test_mini/pub_000654/glare.png \
  --checkpoint weights/checkpoint_best.pt \
  --output examples/predict/pred.png
```

The prediction is saved to `examples/predict/pred.png`. For all available options, run `python infer.py --help`.
