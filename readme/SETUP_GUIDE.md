# Voice Command & Speaker Identification — Integration Guide

This completes the bonus deliverables: **real-time microphone prediction** and
a **Streamlit GUI**, wired up to the Arduino hardware.

## Architecture

```
 [Browser Mic] --or--> [Uploaded File]
        |
        v
  Streamlit App (streamlit_app.py)
   - librosa: trims silence, extracts 17 MFCCs (same as process.ipynb)
   - scaler.pkl -> StandardScaler.transform
   - command_model.pkl -> predicts ON / OFF
   - person_model.pkl  -> predicts speaker name
        |
        v  (Serial, "Person,COMMAND\n")
   Arduino Uno (arduino_voice_control.ino)
   - Parses the line
   - LCD line 1: speaker name, line 2: command
   - LED: ON -> HIGH, OFF -> LOW
```

## 1. Hardware wiring

See `circuit_diagram.png`. Using an **I2C-backpack LCD** instead of a raw
parallel 16x2 LCD cuts the wiring down to 4 wires and avoids needing a
contrast-adjust potentiometer.

| Component      | Pin       | Arduino Uno |
|-----------------|-----------|-------------|
| LCD (I2C)       | GND       | GND         |
| LCD (I2C)       | VCC       | 5V          |
| LCD (I2C)       | SDA       | A4          |
| LCD (I2C)       | SCL       | A5          |
| LED anode (+)   | via 220Ω resistor | D8   |
| LED cathode (-) | —         | GND         |

If you only have a standard parallel 16x2 LCD (no I2C backpack), that's fine
too — just use the classic 6-wire `LiquidCrystal` wiring (RS, E, D4–D7,
contrast pot on V0) instead of `LiquidCrystal_I2C`, and update the two LCD
lines in `setup()`/`handlePrediction()` accordingly. The I2C version is
recommended because it needs far fewer wires and no potentiometer.

## 2. Arduino side

1. Open `arduino_voice_control.ino` in the Arduino IDE.
2. Install the **LiquidCrystal_I2C** library (Library Manager → search
   "LiquidCrystal I2C" by Frank de Brabander).
3. Upload to the Uno. Leave the Serial Monitor **closed** once you run
   Streamlit — only one program can hold the serial port at a time.
4. On boot the LCD shows "Voice ID System / Waiting...".

## 3. Python / Streamlit side

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Make sure `command_model.pkl`, `person_model.pkl`, and `scaler.pkl` sit in
the same folder as `streamlit_app.py`.

In the app:
1. **Sidebar** → pick the Arduino's COM port (e.g. `COM5` on Windows, or
   `/dev/ttyUSB0` / `/dev/cu.usbmodemXXXX` on Linux/Mac) → **Connect**.
2. **Real-time Microphone tab** → click record, say "ON" or "OFF" → the app
   predicts speaker + command and forwards it to the Arduino automatically.
3. **Upload Audio File tab** → for testing with pre-recorded `.wav`/`.ogg`
   clips.
4. The **Prediction History** table at the bottom logs every run for your
   report/demo.

## 4. Notes on the current models

- The person model currently recognizes 2 speakers (`Esraa`, `Wahban`) — add
  more people by following the `Data/<Person>/ON|OFF` structure in the
  README and re-running `process.ipynb`, then re-export the 3 `.pkl` files.
- Feature extraction in the app exactly mirrors `process.ipynb` (17 MFCCs,
  22050 Hz, `top_db=30` silence trimming, MFCC normalization, mean-pooled
  across time) so live predictions stay consistent with training.
