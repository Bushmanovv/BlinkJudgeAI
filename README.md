# BlinkBattle — AI-Powered Staring Contest 🚫👁

**BlinkBattle** is an AI-judged, real-time multiplayer staring contest game powered by OpenCV, MediaPipe, and face landmark tracking. Up to 4 players can face off using a single camera, with blinking detected through Eye Aspect Ratio (EAR) analysis. The last player to blink wins. AI tracks each player's eyes and head position, placing names and real-time status indicators directly above their faces.

---

## ✨ Features

* 👁 Blink detection using EAR and MediaPipe face mesh
* ⚖️ AI acts as an unbiased referee for 1-4 players
* 🧵 Real-time overlay of player names and statuses
* 🚮 Automatic disqualification upon blink detection
* 🔢 Adaptive calibration per player for more accurate EAR thresholds
* 🔄 Dynamic UI updates and status messages
* 📹 Multi-camera support with auto-detection and fallback
* 🔢 EAR tracking, smoothing, and visual feedback

---

## 🎓 How It Works

* **EAR (Eye Aspect Ratio)** is computed per player from facial landmarks
* A player is marked as "blinked" if their EAR drops below a calibrated threshold
* Each frame is processed for face mesh detection and individual eye tracking
* The game ends when only one player remains (or none, if all blinked)
* Calibration phase determines personalized baseline EAR values

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/BlinkBattle.git
cd BlinkBattle
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python
mediapipe
numpy
scipy
```

3. **Run the game:**

```bash
python3 staring_contest.py
```

---

## 🎮 Controls & Usage

* When prompted, enter the number of players (1-4)
* Optionally input custom names
* Players should face the camera during calibration
* Press `SPACE` to start the contest after calibration
* Press `Q` to quit the game

### UI Overlay

| Color | Meaning         |
| ----- | --------------- |
| Green | Active Player   |
| Gray  | Lost / Blinking |
| Red   | Not Detected    |

---

## 👤 Author

**Karim Dwikat**
Computer Vision Developer & AI Enthusiast
Built using Python, OpenCV, MediaPipe, and SciPy

---

## 📄 License

Released under the **MIT License**. See the `LICENSE` file for details.
