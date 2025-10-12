#define LED1 4 
#define LED2 5
#define LED3 6
#define LED4 7
#define LED5 15
#define LED6 16
#define LED7 17
#define OUTPUT1 8//donvi
#define OUTPUT2 18//chuc
//hham bitRead(bit, TT); BIT: 1001
// Thời điểm cuối mỗi lần đổi trạng thái
int loaiden=1;
int pin[7]={LED1,LED2,LED3,LED4,LED5,LED6,LED7};
const byte so[10]=
{
  B0000001,
  B1001111,
  B0010010,
  B0000110,
  B1001100,
  B0100100,
  B0100000,
  B0001111,
  B0000000,
  B0000100
}; 
void seven_led(byte bit,int loaiden)
{
 for(int i=0;i<7;i++) digitalWrite(pin[i],loaiden==1 ? bitRead(bit,i): !bitRead(bit,i));
  
}
void setup() {
  for(int i=0;i<7;i++)
  {
    pinMode(pin[i],OUTPUT);
  }
  pinMode(OUTPUT1,OUTPUT);
  pinMode(OUTPUT2,OUTPUT);
  digitalWrite(OUTPUT1,HIGH);
  digitalWrite(OUTPUT2,HIGH);
}

void loop() {
  for(int i=0;i<=99;i++)
  {
    int hangchuc=i/10;
    int donvi=i%10;
        digitalWrite(OUTPUT1,LOW);
        digitalWrite(OUTPUT2,HIGH);
        seven_led(so[hangchuc],loaiden);

    delay(20);
    digitalWrite(OUTPUT2,LOW);    
    digitalWrite(OUTPUT1,HIGH);
        seven_led(so[donvi],loaiden);
    delay(20);
  }
}
