#define TRIG 13
#define ECHO 14

void setup() {
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  Serial.begin(115200);
}

void loop() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH);

  float distanceCm = (duration * 0.034) / 2;

  Serial.print("Khoang cach la: ");
  Serial.print(distanceCm);
  Serial.println(" cm");

  delay(1000);
}