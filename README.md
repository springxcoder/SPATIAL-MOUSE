# SPATIAL-MOUSE
A SYSTEM THAT TRACKS YOUR HAND AND YOU CAN THEN CONTROL THE CURSOR OF YOU COMPUTER USING THE GESTURES MENTIONED INSIDE THE FILE.

## Features
* Smooth tracking physics.
* Lightweight background execution.

## How to Run This Project: 

### Quick Start & Installation

This universal version runs on **Windows, macOS, and Linux**, and supports **all versions of Python (including 3.12, 3.13, and 3.14+)** by utilizing a pure-Python edge inference framework.

Follow these steps in your terminal or command prompt to get up and running:

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPO_NAME
```

### 2. Set Up a Virtual Environment
Isolate your dependencies to prevent version conflicts with your global system setup.

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 3. Install Core Dependencies
Once your environment is active, upgrade `pip` and install the universal computer vision and automation requirements:
```bash
pip install --upgrade pip
pip install ultralytics opencv-python pyautogui
```

### 4. Run the Air Mouse
Execute the script to start the spatial tracking engine. On its very first run, it will automatically pull down the micro-pose layout asset (approx. 2MB) into your local folder.
```bash
python air_mouse.py
```

### Configuration & OS Permissions Note
* **macOS Users:** You must grant **Accessibility** and **Input Monitoring** privileges to your Terminal or Code Editor (e.g., VS Code) under *System Settings -> Privacy & Security* so `pyautogui` can control the global cursor.
* **Linux Users:** Depending on your window manager distribution, you may need to install `scrot` or switch to X11 if native cursor automation is blocked by Wayland.

*(Note: Replace `main.py` with the actual name of your Python file if it is named differently!)*

## Built With
* Python
