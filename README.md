# Knux: AI-Powered Martial Arts Kinematics Engine

**Knux** is a high-performance biomechanical analysis tool designed to provide elite-level coaching feedback through Computer Vision and LLM integration.

## 🚀 Key Technical Features (MVP)
* **Multi-Stage CV Pipeline:** Uses MediaPipe Pose for 3D landmark extraction with temporal smoothing (`deque`) for stable tracking.
* **Heuristic Analysis Engine:** Geometric algorithms to detect technical flaws like "Chin Exposure" and "Elbow Flare."
* **AI Coaching Bridge:** Pips structured kinematic data into LLM APIs (Gemini/GPT-4) to generate natural language training tips.
* **Decoupled Architecture:** Built with a FastAPI backend designed for asynchronous task processing.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Libraries:** OpenCV, MediaPipe, NumPy
* **Backend:** FastAPI, Uvicorn
* **AI:** Google Gemini API (Flash 1.5)

## 📦 Installation & Setup
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/knux.git`
2. Initialize venv: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install mediapipe opencv-python python-dotenv`
4. Run the engine: `python3 backend/pose_engine.py`

*Note: This project is currently in active development (Sprint 1).*
