# Voice Command & Speaker Identification System (AI Pipeline)

An intelligent voice recognition and speaker identification system powered by Machine Learning, built by the **IEEE SSCS AI Team (Level 2)**. The pipeline processes audio from a microphone or uploaded file and predicts two outputs **simultaneously** from the same recording:

1. **Command** — recognizes the spoken voice command (`ON` / `OFF`)
2. **Person Identification** — identifies the speaker's unique voice biometric (voiceprint) for secure access control

Predictions are forwarded over USB serial to an **Arduino Uno**, which displays the speaker and command on a 16×2 LCD and switches an LED on/off accordingly.

---

## Features

- **Two models, one feature vector** — a single 17-dimensional MFCC feature vector feeds two independent classifiers:
  - `command_model.pkl` → predicts `ON` / `OFF`
  - `person_model.pkl` → predicts the enrolled speaker (`Esraa`, `Wahban` — easily extended)
- **Real-time microphone tab** — record directly in the browser (`audio-recorder-streamlit`)
- **File-upload tab** — test with pre-recorded `.wav` / `.ogg` / `.flac` clips
- **Arduino integration** — automatic serial link (`Person,COMMAND` protocol), with connect/disconnect controls in the sidebar
- **Confidence breakdown** — per-class probability bars for every prediction
- **Prediction history** — every run is logged with timestamp, speaker, command, and confidence for demos/reports
- **No-code enrollment** — add a new speaker just by creating folders and dropping in `.ogg` files (see below)

## How It Works

```
 [Browser Mic] --or--> [Uploaded File]
        |
        v
  Streamlit App (streamlit_app.py)
   - librosa: loads @ 22,050 Hz, trims silence (top_db=30)
   - 17 normalized MFCCs, mean-pooled over time  (same as process.ipynb)
   - scaler.pkl (StandardScaler) -> transform
   - command_model.pkl -> ON / OFF
   - person_model.pkl  -> speaker name
        |
        v  (Serial: "Person,COMMAND\n")
  Arduino Uno (arduino_voice_control.ino)
   - Parses the line
   - LCD line 1: speaker name, line 2: command
   - LED on D8: ON -> HIGH, OFF -> LOW
```

### Training pipeline (`process.ipynb`)

1. **Data collection** — walks `Data/<Person>/ON|OFF/*.ogg`
2. **Preprocessing** — resample to 22,050 Hz, mono; trim silence (`top_db=30`)
3. **Features** — 17 MFCCs (`librosa.feature.mfcc`), normalized with `librosa.util.normalize`, mean-pooled across time → 17 features per clip
4. **Scaling** — `StandardScaler` (saved as `scaler.pkl`)
5. **Models** — two **Random Forest** classifiers (command + person), saved as `command_model.pkl` and `person_model.pkl` with `joblib`
6. **Evaluation** — the notebook reports accuracy / classification reports per model; re-run it after adding data

The Streamlit app's feature extraction **exactly mirrors** the notebook, so live predictions stay consistent with training.

## Dataset

Two speakers currently enrolled, recorded via Telegram voice messages (`.ogg`):

```
Data/
├── Esraa/
│   ├── ON/    <-- ~33 .ogg files
│   └── OFF/   <-- ~40 .ogg files
└── Wahban/
    ├── ON/
    └── OFF/
```

### How to Add a New Speaker (no code changes)

1. **Record audio** — record clips in the default Telegram voice format (`.ogg`)
2. **Create folders** — add them under `Data/` exactly like this:
   ```text
   Data/
   └── [New_Person_Name]/
       ├── ON/   <-- place .ogg files for the turn-on command here
       └── OFF/  <-- place .ogg files for the turn-off command here
   ```
3. **Re-train** — open and run `process.ipynb`, then re-export `command_model.pkl`, `person_model.pkl`, and `scaler.pkl` next to `streamlit_app.py`

## Repository Structure

```
├── Data/                          # enrolled speakers' audio (.ogg)
├── readme/
│   ├── SETUP_GUIDE.md             # wiring diagram + step-by-step integration guide
│   └── requirements.txt
├── arduino_voice_control.ino      # Arduino sketch (LCD + LED, serial protocol)
├── process.ipynb                  # training notebook (features + Random Forest models)
├── streamlit_app.py               # inference app (mic + upload + Arduino link)
├── command_model.pkl              # trained ON/OFF classifier
├── person_model.pkl               # trained speaker-ID classifier
├── scaler.pkl                     # fitted StandardScaler
└── README.md
```

## Installation & Usage

```bash
pip install -r readme/requirements.txt
streamlit run streamlit_app.py
```

Make sure `command_model.pkl`, `person_model.pkl`, and `scaler.pkl` are in the same folder as `streamlit_app.py`.

In the app:
1. **Sidebar** → enter the Arduino's serial port (e.g. `COM5`, `/dev/ttyUSB0`) → **Connect**
2. **Real-time Microphone tab** → click record, say *"ON"* or *"OFF"* → the speaker + command are predicted and sent to the Arduino automatically
3. **Upload Audio File tab** → test with pre-recorded clips
4. Scroll down for the **Prediction History** log

> The app runs fine without an Arduino — predictions and confidence bars still work; only the hardware forwarding is skipped.

## Arduino Integration

The Arduino sketch receives lines in the format `PersonName,COMMAND\n` (e.g. `Esraa,ON`), shows the speaker and command on a 16×2 LCD, and drives an LED (ON → HIGH, OFF → LOW). Full wiring table, library requirements (LiquidCrystal_I2C), and troubleshooting are in **[readme/SETUP_GUIDE.md](readme/SETUP_GUIDE.md)**.

Quick wiring summary (I2C LCD):

| Component | Pin | Arduino Uno |
|---|---|---|
| LCD (I2C) | GND / VCC / SDA / SCL | GND / 5V / A4 / A5 |
| LED anode (+) | via 220Ω resistor | D8 |
| LED cathode (−) | — | GND |

## Tech Stack

- **Python** — `librosa` (audio features), `scikit-learn` (Random Forest, StandardScaler), `joblib` (model persistence)
- **Streamlit** — GUI, `audio-recorder-streamlit` for in-browser mic capture
- **pyserial** — USB link to Arduino
- **Arduino** — LCD + LED output, custom `Person,COMMAND` serial protocol

## Team

IEEE SSCS AI Team — Level 2, first project (bonus: real-time mic + GUI + Arduino integration).
