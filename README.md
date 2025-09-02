# honglab-hand-demo

A mechanical hand demo project for HongLab, Tsinghua University.

## Repository Structure
- `docs/` — documentation (design notes, wiring diagrams)
- `python/` — Python code for ECoG decoding & BLE control
- `firmware/` — STM32 firmware (C code, PlatformIO project)
- `.github/` — GitHub workflows (CI/CD automation)

## Quick Start
### Python
```bash
cd python
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install bleak
python app.py