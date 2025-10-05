# Ghoul Fluids 👻

Real-time, GPU-driven **stable fluids** that respond to a live **people segmentation mask**.  
Built with **ModernGL + GLFW + OpenCV + MediaPipe**. Optional split-screen shows the camera feed next to the effect — perfect for tuning a Halloween display.

<p align="center">
  <img src="./screenshot.png" alt="Ghoul Fluids screenshot" width="900">
</p>


---

## Features

- **Stable fluids (Stam-style)** on the GPU (advect, vorticity, pressure projection)
- **Edge-based forces** from a live segmentation mask (pushes dye along silhouettes)
- **Split-screen** option: left = camera, right = fluid (`--split`)
- **No giant monolith**: clean module layout, easy to extend
- **Headless tests** (pytest) and **GitHub Actions** CI

---

## Quick start

```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Editable install (includes the ghoulfluids CLI)
pip install -e .

# 3) Run
ghoulfluids            # fullscreen fluid
ghoulfluids --split    # split view (camera | fluid)
