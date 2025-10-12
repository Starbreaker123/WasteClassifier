#define LED 2
#define BUT 4

int nowtime=0,currentime=0;
void thaydoi()
{
  digitalWrite(LED,0);
}
void setup() {
  // put your setup code here, to run once:
  pinMode(LED,OUTPUT);
  pinMode(BUT,INPUT);
  attachInterrupt(digitalPinToInterrupt(BUT),thaydoi,HIGH);
}
void loop() {
  // put your main code here, to run repeatedly:
     currentime=millis();
  if(currentime-nowtime>=500)
  {
    digitalWrite(LED,!digitalRead(LED));
    nowtime=currentime;
  }
}
