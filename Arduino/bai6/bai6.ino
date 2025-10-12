#define LED1 12
//println,available,read(doc tung byte trong bo nho dem)
char l[30]="nhap chu";
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println(l);
  pinMode(LED1,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  String a="\0",b,c;
  if(Serial.available())
  {
     
     c=Serial.readString();
    c.trim();
     if(c=="OFF")
     {
        digitalWrite(LED1,0);
     }
     else
     {
      digitalWrite(LED1,1);
     }
  }
  else
  {
    digitalWrite(LED1,digitalRead(LED1));
  }

  }  


