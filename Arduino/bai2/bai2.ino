#define BUT 2
#define LED2 4
int button_current=0,button_last=1,press=0;
int debounce=0,timelast=0,timecurrent=0; 
void setup() {
  // put your setup code here, to run once:
  pinMode(BUT,INPUT);
  pinMode(LED2,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  timecurrent=millis();
  button_current=digitalRead(BUT);
  if(button_current!=button_last)
  {
    if(debounce==0)
    {
      debounce=1;
      timelast=timecurrent;
    }
  }
    if(debounce && (timecurrent-timelast)>=8)
    {
      if(button_current==0)
      {
        digitalWrite(LED2,!digitalRead(LED2));
        press++;
      
      Serial.println(press);
    }
    button_last=button_current;
    debounce=0;
  }
}
