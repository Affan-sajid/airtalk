#include <Wire.h>
#include <Adafruit_ISM330DHCX.h>

#define BUTTON_PIN 1

Adafruit_ISM330DHCX ism;

float accelBiasX = 0, accelBiasY = 0, accelBiasZ = 0;
float gyroBiasX  = 0, gyroBiasY  = 0, gyroBiasZ  = 0;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Wire.begin(8, 9);
  ism.begin_I2C(0x6B, &Wire);

  // Hold the pen still and flat during this window
  Serial.println("CALIBRATING... Hold pen still and flat for 2 seconds.");
  const int SAMPLES = 200;
  for (int i = 0; i < SAMPLES; i++) {
    sensors_event_t accel, gyro, temp;
    ism.getEvent(&accel, &gyro, &temp);
    accelBiasX += accel.acceleration.x;
    accelBiasY += accel.acceleration.y;
    accelBiasZ += (accel.acceleration.z);
    gyroBiasX  += gyro.gyro.x;
    gyroBiasY  += gyro.gyro.y;
    gyroBiasZ  += gyro.gyro.z;
    delay(10);
  }
  accelBiasX /= SAMPLES; accelBiasY /= SAMPLES; accelBiasZ /= SAMPLES;
  gyroBiasX  /= SAMPLES; gyroBiasY  /= SAMPLES; gyroBiasZ  /= SAMPLES;
  Serial.println("READY");
}

void loop() {
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
  Serial.print(rotX * 100, 2); Serial.print(",");
  Serial.print(rotY * 100, 2); Serial.print(",");
  Serial.print(linX * 100, 2); Serial.print(",");
  Serial.print(linY * 100, 2); Serial.print(",");
  Serial.print(penDown); Serial.print(",");
  Serial.println(accelBiasZ * 100, 2);

  delay(10);
}