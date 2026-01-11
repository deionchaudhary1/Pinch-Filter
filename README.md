# Pinch Filter 🎥✋

**Pinch Filter** is a real-time computer vision application built with **Python, OpenCV, and MediaPipe** that allows users to apply dynamic camera distortions using **hand pinch gestures** and **keyboard controls**.

By pinching with both hands, users can define a bounding box on the webcam feed and apply interactive visual effects inside that region.

---

## ✨ Features

- 🖐 Real-time **two-hand pinch detection** using MediaPipe Hands  
- 📦 **Bounding box creation** defined by dual pinch points  
- 🎨 Multiple **camera distortion effects** applied live  
  - Gaussian Blur  
  - Camera-style Focus Blur  
  - Lens Zoom  
  - Swirl Distortion  
- ⌨ Keyboard controls to toggle between effects  
- ⚡ Runs fully in real time using OpenCV  

---

## 📸 How It Works

1. Show **two hands** to the webcam  
2. Pinch thumb and index finger on **both hands**  
3. A bounding box appears between the pinch points  
4. The selected effect is applied inside the box  
5. Switch effects using keyboard shortcuts  

---

## 🧠 Tech Stack

- **Python 3.11**
- **OpenCV**
- **MediaPipe**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pinch-filter.git
cd pinch-filter

Create and activate a virtual environment (recommended):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python pinch_filter.py
```

## Controls and Effects

### Controls
Gesture Controls: Two-hand pinch → Create and control the effect box

#### Key Action

1 -> Gaussian Blur

2 -> Camera Blur

3 -> Lens Zoom

4 -> Swirl

[ -> Previous effect

] -> Next effect

ESC -> Exit application

### Effects
Gaussian Blur: Applies a strong Gaussian blur to smooth the selected region.

Camera Blur: Simulates a camera focus effect with a soft transition into the blurred area.

Lens Zoom: Magnifies the contents of the box, mimicking a zoom lens effect.

Swirl: Applies a rotational swirl distortion centered within the bounding box.
