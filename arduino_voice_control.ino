/*
  Voice Command & Speaker Identification System — Arduino Integration
  IEEE SSSC AI Team — First Project

  HOW IT WORKS
  ------------
  The ML models run on the laptop (inside the Streamlit app / Python script).
  Python predicts the SPEAKER and the COMMAND from a recorded audio clip,
  then sends a single line over USB Serial in the format:

      PersonName,COMMAND\n

  Examples:
      Esraa,ON\n
      Wahban,OFF\n

  This sketch reads that line, shows the speaker's name and the command
  status on a 16x2 I2C LCD, and switches the LED ON or OFF accordingly.

  WIRING (see circuit_diagram.png)
  ---------------------------------
  LCD (I2C backpack)      Arduino Uno
    GND ------------------- GND
    VCC ------------------- 5V
    SDA ------------------- A4
    SCL ------------------- A5

  LED (with 220 ohm resistor in series)
    Anode  (+) ------------ D8 (through resistor)
    Cathode(-) ------------ GND

  LIBRARIES REQUIRED (Arduino IDE > Library Manager)
    - LiquidCrystal_I2C  (by Frank de Brabander, or "LiquidCrystal I2C")
    - Wire (built-in)
*/

#include <LiquidCrystal.h>
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int LED_PIN = 8;
const long SERIAL_BAUD = 9600;

String lastPerson = "";
String lastCommand = "";
unsigned long lastMessageTime = 0;
const unsigned long IDLE_TIMEOUT_MS = 8000; // revert to idle screen after 8s

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  lcd.begin(16, 2);
  showIdleScreen();
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() > 0) {
      int commaIndex = line.indexOf(',');
      if (commaIndex > 0) {
        String person = line.substring(0, commaIndex);
        String command = line.substring(commaIndex + 1);
        command.trim();
        person.trim();
        command.toUpperCase();

        handlePrediction(person, command);
      }
    }
  }
}

void handlePrediction(String person, String command) {
  lastPerson = person;
  lastCommand = command;
  lastMessageTime = millis();

  // Update LED
  if (command == "ON") {
    digitalWrite(LED_PIN, HIGH);
  } else if (command == "OFF") {
    digitalWrite(LED_PIN, LOW);
  }

  // Update LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Speaker: " + truncate(person, 7));
  lcd.setCursor(0, 1);
  lcd.print("Command: " + command);

  // Echo back for debugging / confirmation in the Streamlit app
  Serial.print("ACK,");
  Serial.print(person);
  Serial.print(",");
  Serial.println(command);
}

void showIdleScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Voice ID System");
  lcd.setCursor(0, 1);
  lcd.print("Waiting...");
}

String truncate(String s, int maxLen) {
  if (s.length() <= maxLen) return s;
  return s.substring(0, maxLen);
}
