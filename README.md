# Open-rppg — macOS Branch

> **This is a fork of [open-rppg](https://github.com/KegangWangCCNU/open-rppg) by KegangWangCCNU**, further branched from a classmate's adaptation for our university project.
> This branch contains small compatibility tweaks to make the existing code run on macOS (Apple Silicon & Intel) alongside Windows. No algorithms, models, or core logic were changed — all credit for the actual work goes to the original authors.

Open-rppg is a Python toolbox for Remote Photoplethysmography (rPPG) inference, providing a unified interface for several state-of-the-art deep learning models to measure physiological signals (heart rate, HRV) from facial video. It supports both offline video processing and low-latency real-time inference using JAX.

> Original API documentation: [https://kegangwangccnu.github.io/open-rppg/](https://kegangwangccnu.github.io/open-rppg/)

<img width="948" height="250" alt="image" src="https://github.com/user-attachments/assets/5c945368-67bf-4ccb-8822-fa359a787cdd" />

***

## What This Branch Changes

These are the only changes made here — small compatibility tweaks so the existing code runs on macOS without touching any algorithms or models.

| File | Change |
| :--- | :--- |
| `main.py` | Camera backend auto-detection: `CAP_AVFOUNDATION` on macOS, `CAP_DSHOW` on Windows, `CAP_ANY` elsewhere |
| `main.py` | `ort.InferenceSession` now passes explicit `providers=['CPUExecutionProvider']` to avoid ONNX warnings on macOS |
| `main.py` | `cv2.CAP_PROP_ORIENTATION_META` wrapped in `getattr` fallback for older OpenCV builds on macOS |
| `pyproject.toml` | `setuptools` pinned to `<70` to keep `pkg_resources` working (see note below) |
| `.vscode/settings.json` | `PYTHONPATH` set for Windows, macOS, and Linux terminals |
| `.vscode/launch.json` | `JAX_PLATFORMS=cpu` added to prevent a JAX Metal crash on Apple Silicon |

***

## ⚠️ setuptools Version Requirement

**`setuptools` must stay below version 70.** Both this project and its dependency `heartpy` rely on `pkg_resources`, which was removed as a default importable module in `setuptools>=70`. If you ever see this error:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

just run:

```bash
pip install "setuptools<70"
```

The `pyproject.toml` in this branch already pins `"setuptools>=61.0,<70"`, so a fresh `pip install -e .` should handle this automatically. It's worth knowing about in case you run into it anyway — for example, if another tool in your environment upgrades setuptools without you expecting it.

> **On Apple Silicon and `jax-metal`:** The `metal` optional dependency exists in `pyproject.toml` but is currently broken — it crashes with `UNIMPLEMENTED: default_memory_space` at import time. The `.env` file forces `JAX_PLATFORMS=cpu` to bypass this. CPU JAX is fast enough for real-time RPPG on M-series chips, so this isn't really a limitation in practice.

***

## Getting Started

### macOS (Apple Silicon & Intel)

```bash
# 1. Clone and enter the project
git clone <your-fork-url>
cd open-rppg

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (setuptools<70 is handled automatically)
pip install -e .

# 4. Create a .env file in the project root
echo "KERAS_BACKEND=jax" >> .env
echo "JAX_PLATFORMS=cpu" >> .env

# 5. Verify everything works
KERAS_BACKEND=jax JAX_PLATFORMS=cpu python3 -c "from rppg import Model; print('Import OK!')"
```

Then open the project in VS Code and press **F5** — `launch.json` is already set up correctly.

> **Camera access:** macOS will ask for camera permissions on first run. Click Allow. If you missed it, go to **System Settings → Privacy & Security → Camera** and enable access for Terminal or VS Code.

***

## Known Warnings You Can Ignore

These show up on macOS but don't affect anything:

| Warning | Why it appears |
| :--- | :--- |
| `pkg_resources is deprecated` | setuptools 69.x deprecation notice — harmless |
| `AVFFrameReceiver implemented in both...` | `av` and `cv2` both bundle `libavdevice` on macOS — harmless for webcam use |
| `Platform METAL is experimental` | `jax-metal` is installed but `JAX_PLATFORMS=cpu` overrides it — CPU is used as intended |

***

## Usage

### Process a video file

```python
import rppg

model = rppg.Model()
results = model.process_video("path/to/video.mkv")
print(f"Heart rate: {results['hr']} BPM")
```

**Result keys:** `hr`, `SQI` (signal quality 0–1), `latency`, `hrv` (dict with `bpm`, `ibi`, `sdnn`, `rmssd`, `pnn50`, `LF/HF`, `breathingrate`)

***

### Real-time webcam inference

```python
import rppg, time, cv2

model = rppg.Model()
with model.video_capture(0):
    last_process_time = 0
    current_hr = None

    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        now = time.time()
        if now - last_process_time > 1.0:
            result = model.hr(start=-10)
            if result and result['hr']:
                current_hr = result['hr']
                print(f"HR: {current_hr:.1f} BPM")
            last_process_time = now

        if box is not None:
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if current_hr is not None:
                cv2.putText(frame, f"HR: {current_hr:.1f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("rPPG Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

***

### Other useful methods

```python
# Raw BVP waveform
bvp, timestamps = model.bvp()

# Specific time window
bvp_slice, ts_slice = model.bvp(start=10, end=20)
metrics = model.hr(start=-15)  # last 15 seconds

# Use a different model architecture
model = rppg.Model('PhysMamba.pure')

# Pass frames directly as a numpy array (T, H, W, 3) uint8
result = model.process_video_tensor(video_tensor, fps=30.0)
result = model.process_faces_tensor(faces_tensor, fps=30.0)
```

***

## License

Source code is released under the **MIT License**. Pretrained model weights are the intellectual property of their respective authors — see the original publications for terms.

***

## Credits & Citation

All the actual research and engineering is from the original open-rppg project and the papers below. If you use this in your own work, please cite them:

```bibtex
@article{wang2025memory,
  title={Memory-efficient Low-latency Remote Photoplethysmography through Temporal-Spatial State Space Duality},
  author={Wang, Kegang and Tang, Jiankai and Fan, Yuxuan and Ji, Jiatong and Shi, Yuanchun and Wang, Yuntao},
  journal={arXiv preprint arXiv:2504.01774},
  year={2025}
}

@inproceedings{luo2024physmamba,
  title={PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba},
  author={Luo, Chaoqi and Xie, Yiping and Yu, Zitong},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={248--259},
  year={2024},
  organization={Springer}
}

@inproceedings{zou2025rhythmmamba,
  title={RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement},
  author={Zou, Bochao and Guo, Zizheng and Hu, Xiaocheng and Ma, Huimin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={11077--11085},
  year={2025}
}

@inproceedings{yu2022physformer,
  title={PhysFormer: Facial Video-Based Physiological Measurement with Temporal Difference Transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip HS and Zhao, Guoying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4186--4196},
  year={2022}
}

@article{liu2020multi,
  title={Multi-task temporal shift attention networks for on-device contactless vitals measurement},
  author={Liu, Xin and Fromm, Josh and Patel, Shwetak and McDuff, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={19400--19411},
  year={2020}
}

@inproceedings{liu2023efficientphys,
  title={EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement},
  author={Liu, Xin and Hill, Brian and Jiang, Ziheng and Patel, Shwetak and McDuff, Daniel},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5008--5017},
  year={2023}
}

@article{yu2019remote,
  title={Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks},
  author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
  journal={arXiv preprint arXiv:1905.02419},
  year={2019}
}
```