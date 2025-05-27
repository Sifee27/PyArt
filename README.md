# 🎨 PyArt — Real-Time ASCII Art with Gesture Control

**PyArt** is an interactive Python tool that transforms **images** and **live webcam feeds** into stunning ASCII art — with **gesture-based controls** for hands-free interactivity. Built for fun, learning, and visual experimentation using Python and computer vision!

---

## ✨ Features

- 🖼️ **Image to ASCII**: Upload any image and see it rendered as ASCII.
- 🎥 **Live Webcam to ASCII**: View yourself as ASCII in real time.
- ✋ **Gesture-Based Controls** using your webcam:
  - 👍 **Thumbs Up** → Increase detail (higher ASCII resolution).
  - 👎 **Thumbs Down** → Decrease detail (lower ASCII resolution).
  - ✊ **Closed Fist** → Take a snapshot and save it.
- 🌈 **Colored ASCII Output**: Displayed in terminal using ANSI color codes.
- 💾 **Save Snapshots**: Captures webcam ASCII output to `.txt` file.
- 🔁 **Real-time Processing**: Fast, responsive updates as you move and gesture.

---

## 🛠️ Tech Stack

- **Python 3.x**
- [`OpenCV`](https://opencv.org/) — For capturing webcam input and handling video frames
- [`MediaPipe`](https://developers.google.com/mediapipe) — For real-time hand gesture detection
- [`Pillow`](https://python-pillow.org/) — For loading and manipulating image files
- [`NumPy`](https://numpy.org/) — For efficient image data handling and transformations
- [`colorama`](https://pypi.org/project/colorama/) — For adding color to terminal output

---

## 🎮 Controls & Gestures

| Gesture | Action |
|--------|--------|
| 👍 Thumbs Up | Increase ASCII resolution (more detail) |
| 👎 Thumbs Down | Decrease ASCII resolution (less detail) |
| ✊ Closed Fist | Capture and save current frame |

---

## 📦 Installation

```bash
git clone https://github.com/sifee27/pyart
cd pyart

pip install -r requirements.txt
🧪 Future Improvements
🖼️ Drag-and-drop GUI with live preview

🌐 Web version via Flask or FastAPI

🎛️ More gestures for switching filters and color schemes

🎥 Record webcam ASCII sessions as video

✏️ ASCII "drawing mode" with webcam and gestures

🔗 Why PyArt?
Built for the Athena Hack Club Hackathon, PyArt is a unique blend of:

💡 Creative coding

🧠 Computer vision & gesture recognition

🎨 ASCII aesthetics

🛠️ Real-world Python development

👩‍💻 Contributions Welcome
Want to help add new features or gestures? PRs are open!

