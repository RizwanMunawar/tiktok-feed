# Speeding Up Ultralytics YOLO Without Losing Accuracy

A practical, field-tested guide to making Ultralytics YOLO models (YOLOv5 →
YOLO11 → YOLO26) run faster **while keeping mAP essentially unchanged**.

The short answer to "can I make it faster without losing accuracy?" is **yes**.
Most of the speed on the table comes from *how* you run and deploy the model,
not from shrinking the network. The two biggest levers — exporting to an
optimized runtime (TensorRT / ONNX / OpenVINO) and running in **FP16** — are
effectively **lossless** (mAP within ~0.5 of the PyTorch baseline). Techniques
that *do* cost accuracy (INT8, smaller models, lower resolution) are called out
explicitly so you can avoid them or use them with guardrails.

---

## TL;DR — the decision table

| Goal / hardware | Do this | Speedup | Accuracy cost |
|---|---|---|---|
| **NVIDIA GPU** (server, Jetson) | Export to **TensorRT** `format="engine"` + **FP16** (`quantize=16`) | up to ~5× vs PyTorch GPU | **≈ none** (verify mAP within 0.5) |
| **Intel/AMD CPU** | Export to **OpenVINO** or **ONNX Runtime**, FP32 | up to ~3× vs PyTorch CPU | **none** |
| **CPU, need more** | OpenVINO **INT8** (`quantize=8` + calibration data) | additional ~1.5–2× | small (~0.5–3% mAP) |
| **Any GPU, no re-export** | Run inference with **`half=True`** | ~1.5–2× | **≈ none** |
| **Video streams** | `stream=True`, tune `vid_stride`, right `imgsz`, `batch` | large, free | none |
| **Absolute lowest latency** | TensorRT **INT8** (`quantize=8` + calibration) | up to ~8× | ~1 mAP (guard it) |

> Rule of thumb: **FP16 is free accuracy. INT8 is a trade.** Prefer FP16 first,
> and only reach for INT8 when you've measured that the accuracy hit is
> acceptable for your task.

---

## 1. Understand where the time goes

A YOLO `predict` call is three stages:

1. **Preprocess** — resize/letterbox, BGR→RGB, HWC→CHW, normalize, to-tensor.
2. **Inference** — the forward pass through the network.
3. **Postprocess** — decode boxes, confidence filtering, **NMS**.

`results[0].speed` reports all three (`preprocess`, `inference`, `postprocess`
in ms). Always look at this before optimizing — if 40% of your latency is
preprocess or NMS, exporting to TensorRT won't help those stages much, but
`imgsz`, `max_det`, `conf`, and `batch` will. Measure first, optimize the
actual bottleneck.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
r = model("bus.jpg")
print(r[0].speed)  # {'preprocess': .., 'inference': .., 'postprocess': ..} ms
```

---

## 2. Zero-cost runtime settings (no re-export, no accuracy loss)

These change *how you call* the model. None of them alter the weights, so mAP is
untouched (unless noted).

| Setting | What it does | Accuracy impact |
|---|---|---|
| `half=True` | FP16 inference on a CUDA GPU — ~1.5–2× faster, half the memory | **≈ none** on modern GPUs |
| `imgsz` | Match to your data; don't infer at 1280 if you trained at 640 | none if matched to training |
| `batch=N` | Batch frames/images to saturate the GPU | none |
| `stream=True` | Generator over frames — constant memory on long videos | none (prevents OOM) |
| `vid_stride=N` | Process every Nth video frame | none per-frame; you skip frames |
| `max_det` | Cap detections (default 300) — less NMS/postprocess work | none for normal scenes |
| `conf` / `iou` | Higher `conf` filters earlier; tuned `iou` reduces NMS work | tune on your val set |
| `augment=False` | Keep test-time augmentation **off** (it's off by default) | TTA is slower; off = faster |
| `device=0` | Pin to GPU explicitly | none |
| `retina_masks=False` | Skip high-res mask upsampling (segmentation) | none for most uses |

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# GPU, half precision, batched, streaming over a video — all lossless
results = model.predict(
    source="traffic.mp4",
    device=0,
    half=True,       # FP16 — big win, negligible accuracy change on GPU
    imgsz=640,       # match training resolution
    stream=True,     # memory-efficient generator for long video
    vid_stride=1,    # raise to 2–3 if you can tolerate skipping frames
    max_det=100,     # cap if your scenes never have 300 objects
)
for r in results:
    boxes = r.boxes
```

**Layer fusion** (Conv+BN) is applied automatically by Ultralytics on load
(`model.fuse()`), so you already get that speedup for free.

**`torch.compile`** (PyTorch 2.x) can give an extra ~10–30% on GPU after a
one-time warmup compile, with no accuracy change. Warm up with a couple of dummy
inferences before timing so the compiled kernels are cached.

---

## 3. The big lever: export to an optimized runtime

This is where the real gains are, and the accuracy-preserving ones (FP32/FP16)
are effectively lossless because they don't change the numerics meaningfully —
they change the *execution engine* (kernel fusion, auto-tuning, memory
planning).

### Pick the format for your target

| Target hardware | Best format | Precision to start with |
|---|---|---|
| NVIDIA GPU / Jetson | `engine` (**TensorRT**) | FP16 (`quantize=16`) |
| Intel / AMD CPU | `openvino` or `onnx` | FP32, then INT8 if needed |
| Apple silicon | `coreml` | FP16 |
| Android / ARM / edge | `tflite`, `ncnn`, `rknn`, `mnn` | FP16, INT8 for NPUs |
| Portable / broad | `onnx` (ONNX Runtime) | FP32 / FP16 |

### The `quantize` argument (current API)

Recent Ultralytics replaces the old `half=True` / `int8=True` **export** flags
with a single `quantize` argument (the old flags still work but emit a
deprecation warning):

- `quantize=32` → FP32 (full precision, lossless)
- `quantize=16` → FP16 (near-lossless, the recommended fast default)
- `quantize=8`  → INT8 (fastest; **requires calibration data**, costs accuracy)
- `"w8a16"`, `"w8a32"` → mixed weight/activation schemes

> For **runtime** `predict()` you still pass `half=True` — the deprecation only
> applies to the `export()` precision flags.

### TensorRT (NVIDIA GPU) — FP16, effectively lossless

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(
    format="engine",   # TensorRT
    imgsz=640,
    quantize=16,       # FP16 — recommended
    dynamic=True,      # allow variable batch/size
    batch=8,           # max batch the engine will support
    workspace=4,       # GB of build scratch memory
)

# Inference is identical — just point YOLO at the engine
trt = YOLO("yolo11n.engine")
trt("bus.jpg", half=True)
```

**Measured TensorRT results (YOLO detection on COCO):**

*NVIDIA A100:*

| Precision | Inference (ms) | mAP50 | mAP50-95 |
|---|---|---|---|
| FP32 | 0.52 | 0.52 | 0.37 |
| **FP16** | **0.34** | **0.52** | **0.37** ← *no accuracy loss* |
| INT8 | 0.28 | 0.47 | 0.33 ← *~10% mAP drop* |

*NVIDIA RTX 3080:*

| Precision | Inference (ms) | mAP50 | mAP50-95 |
|---|---|---|---|
| FP32 | 1.06 | 0.52 | 0.37 |
| **FP16** | **0.62** | **0.52** | **0.37** |
| INT8 | 0.52 | 0.47 | 0.33 |

The takeaway: **FP16 nearly halves latency with identical mAP.** INT8 is faster
still but drops mAP50-95 from 0.37 → 0.33 here — only worth it when latency is
critical and you've verified the drop is tolerable.

### OpenVINO / ONNX Runtime (CPU) — lossless FP32

For CPU deployment, exporting to OpenVINO or ONNX Runtime gives ~1.6–3× over
PyTorch CPU with **no accuracy change** (FP32):

```python
from ultralytics import YOLO

YOLO("yolo11n.pt").export(format="openvino")        # or format="onnx"
ov = YOLO("yolo11n_openvino_model/")
ov("bus.jpg")
```

**Measured OpenVINO FP32 vs PyTorch (Intel Core Ultra CPU):**

| Model | PyTorch (ms) | OpenVINO FP32 (ms) | Speedup |
|---|---|---|---|
| YOLOn | 29.3 | 14.4 | ~2.0× |
| YOLOs | 59.3 | 35.0 | ~1.7× |
| YOLOm | 143.2 | 90.5 | ~1.6× |

All at **the same mAP** — this is a pure, free win on CPU.

---

## 4. INT8 — the one that costs accuracy (and how to minimize it)

INT8 is the fastest option but it *quantizes activations*, so some accuracy loss
is expected. It is **not** lossless — treat it as a deliberate trade. If your
requirement is strictly "no accuracy loss," **stop at FP16** and skip this
section.

If you do need INT8, minimize the damage:

1. **Always provide representative calibration data** via `data=`. The
   calibration set should look like your production images — same domain,
   lighting, resolution, class balance. Bad calibration is the #1 cause of large
   INT8 drops.
2. **Use enough calibration samples** (a few hundred to ~1k images is typical).
3. **Consider mixed precision** (`"w8a16"`) to keep activations at FP16 where it
   matters.
4. **Always re-validate** and compare against baseline before shipping.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# TensorRT INT8 — needs calibration data
model.export(format="engine", quantize=8, data="coco.yaml",
             imgsz=640, dynamic=True, batch=8, workspace=4)

# OpenVINO INT8 (CPU)
model.export(format="openvino", quantize=8, data="coco8.yaml")
```

With good calibration, OpenVINO INT8 drops can be small (measured examples:
YOLOn −0.75%, YOLOm −2.4% mAP50-95). Whether that's acceptable is your call —
but you must *measure it*, never assume.

---

## 5. Prove you didn't lose accuracy — benchmark & val

Never trust a speedup blindly. Ultralytics ships a **benchmark mode** that
reports mAP, latency, and file size across every format so you can compare
apples-to-apples:

```python
from ultralytics.utils.benchmarks import benchmark

# Benchmarks all formats on GPU, reporting mAP + speed + size for each
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, device=0)
```

```bash
yolo benchmark model=yolo11n.pt data=coco8.yaml imgsz=640 device=0
```

Then validate the exported model directly and diff the mAP against the PyTorch
baseline:

```python
from ultralytics import YOLO

base = YOLO("yolo11n.pt").val(data="coco.yaml")          # baseline mAP
trt  = YOLO("yolo11n.engine").val(data="coco.yaml")      # exported mAP
print(base.box.map, trt.box.map)  # compare mAP50-95
```

**Acceptance guardrails (recommended):**

- **FP16:** expect mAP within **±0.5** of baseline. If it drops more, rebuild
  with more `workspace`, check `imgsz`/`dynamic` settings.
- **INT8:** expect within **±1.0** mAP50 for a well-calibrated model. If worse,
  improve calibration data or fall back to FP16.

---

## 6. Recommended recipes

**Server GPU, "fast but no accuracy loss":**
```python
YOLO("yolo11m.pt").export(format="engine", quantize=16, dynamic=True,
                          batch=16, workspace=4, imgsz=640)
```

**Jetson / edge GPU, latency-critical (accept small drop):**
```python
YOLO("yolo11n.pt").export(format="engine", quantize=8, data="my_data.yaml",
                          imgsz=640, workspace=2)
```

**CPU deployment, no accuracy loss:**
```python
YOLO("yolo11s.pt").export(format="openvino")   # FP32, ~1.7–2× free
```

**Quick GPU win, no export at all:**
```python
YOLO("yolo11n.pt").predict("video.mp4", device=0, half=True,
                           stream=True, imgsz=640)
```

---

## 7. Common pitfalls

- **Inferring at a bigger `imgsz` than you trained.** Doing 1280 on a 640-trained
  model is slower *and* often less accurate. Match training resolution.
- **Not warming up.** The first 1–3 inferences (and the first `torch.compile` /
  TensorRT run) are slow. Discard them when benchmarking.
- **Comparing PyTorch-GPU latency to TensorRT-CPU** or other apples-to-oranges
  setups. Use `yolo benchmark` for fair comparison.
- **Shipping INT8 without re-validating.** Always diff mAP against baseline.
- **Leaving `augment=True`** (test-time augmentation) on in production — it's
  much slower for a tiny accuracy gain that you usually don't need.
- **CPU thread contention.** Set thread counts appropriately; more threads is not
  always faster for a single small model.

---

## Summary

| Lever | Typical gain | Accuracy-safe? |
|---|---|---|
| Right `imgsz`, `batch`, `stream`, `max_det`, `conf`/`iou` | free–large | ✅ yes |
| `half=True` at inference (GPU) | ~1.5–2× | ✅ yes (≈none) |
| Layer fusion + `torch.compile` | ~10–30% | ✅ yes |
| **TensorRT FP16** (GPU) | **up to ~5×** | ✅ **yes (recommended)** |
| **OpenVINO / ONNX** (CPU, FP32) | **up to ~3×** | ✅ **yes** |
| TensorRT / OpenVINO **INT8** | up to ~8× / +1.5–2× | ⚠️ small trade — calibrate & verify |

Start with the free runtime settings, then export to the right runtime in
**FP16** for your hardware. That combination alone typically delivers a 2–5×
speedup with **no measurable accuracy loss** — and you only descend to INT8 when
you've measured that its trade-off is acceptable.

---

## Sources

- [Ultralytics Docs — Model Export](https://docs.ultralytics.com/modes/export)
- [Ultralytics Docs — TensorRT Integration](https://docs.ultralytics.com/integrations/tensorrt)
- [Ultralytics Docs — OpenVINO Integration](https://docs.ultralytics.com/integrations/openvino)
- [Ultralytics Docs — Predict Mode](https://docs.ultralytics.com/modes/predict)
- [Ultralytics Docs — Benchmark Mode](https://docs.ultralytics.com/modes/benchmark)
- [Ultralytics Blog — Optimizing YOLO with TensorRT](https://www.ultralytics.com/blog/optimizing-ultralytics-yolo-models-with-the-tensorrt-integration)
