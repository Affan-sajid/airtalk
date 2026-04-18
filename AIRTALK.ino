#include <Wire.h>
#include <stdio.h>
#include <Adafruit_ISM330DHCX.h>

#define BUTTON_PIN 1

// One buffer + one write per sample avoids USB CDC dropping bytes between many
// Serial.print() calls (shows up on the host as "bad line" / split CSV rows).
static char serialLine[96];

Adafruit_ISM330DHCX ism;

float accelBiasX = 0, accelBiasY = 0, accelBiasZ = 0;
float gyroBiasX  = 0, gyroBiasY  = 0, gyroBiasZ  = 0;

void runCalibration() {
  accelBiasX = 0; accelBiasY = 0; accelBiasZ = 0;
  gyroBiasX  = 0; gyroBiasY  = 0; gyroBiasZ  = 0;

  Serial.println("CALIBRATING... Hold pen still and flat for 2 seconds.");
  const int SAMPLES = 200;
  for (int i = 0; i < SAMPLES; i++) {
    sensors_event_t accel, gyro, temp;
    ism.getEvent(&accel, &gyro, &temp);
    accelBiasX += accel.acceleration.x;
    accelBiasY += accel.acceleration.y;
    accelBiasZ += accel.acceleration.z;
    gyroBiasX  += gyro.gyro.x;
    gyroBiasY  += gyro.gyro.y;
    gyroBiasZ  += gyro.gyro.z;
    delay(10);
  }
  accelBiasX /= SAMPLES; accelBiasY /= SAMPLES; accelBiasZ /= SAMPLES;
  gyroBiasX  /= SAMPLES; gyroBiasY  /= SAMPLES; gyroBiasZ  /= SAMPLES;
  Serial.println("READY");
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Wire.begin(8, 9);
  ism.begin_I2C(0x6B, &Wire);
  runCalibration();
}

void loop() {
  // Re-calibrate on demand when Python sends 'R'
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'R') {
      runCalibration();
      return;
    }
  }

  sensors_event_t accel, gyro, temp;
  ism.getEvent(&accel, &gyro, &temp);

  // Bias-corrected gyro (rad/s) — primary signal for air writing
  float rotX = gyro.gyro.z - gyroBiasZ;   // left / right
  float rotY = gyro.gyro.y - gyroBiasY;   // up   / down

  // Bias-corrected linear acceleration (m/s²)
  float linX = accel.acceleration.x - accelBiasX;
  float linY = accel.acceleration.y - accelBiasY;

  // Button: LOW = pressed = pen down → send 1
  int penDown = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;

  // rotX,rotY,linX,linY,pen,accelBiasZ  (motion *100; biasZ is averaged raw Z at cal, m/s² *100)
  snprintf(
      serialLine,
      sizeof(serialLine),
      "%.2f,%.2f,%.2f,%.2f,%d,%.2f",
      rotX * 100.0f,
      rotY * 100.0f,
      linX * 100.0f,
      linY * 100.0f,
      penDown,
      accelBiasZ * 100.0f);
  Serial.println(serialLine);
  Serial.flush();

  delay(10);
}