#define POWER 1
#define AO 2
void setup() {
  Serial.begin(19200);
  pinMode(POWER,OUTPUT);
  pinMode(AO,INPUT);
}

void loop() {
  digitalWrite(POWER,1);
  delay(10);
  uint16_t a=analogRead(AO);
  digitalWrite(POWER,0);
  Serial.println(a);
  delay(1000);
}