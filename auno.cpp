    int ledPin = 8;
    void setup() {
    Serial.begin(9600);
    pinMode(ledPin, OUTPUT); 
    digitalWrite(ledPin, LOW);
    }
    void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();
        
        if (command == '1') {
        digitalWrite(ledPin, HIGH);
        } 
        else if (command == '0') {
        digitalWrite(ledPin, LOW);
        }
    }
    }