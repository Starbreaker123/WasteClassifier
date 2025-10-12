/**/
/*************************************************************
  Download latest ERa library here:
    https://github.com/eoh-jsc/era-lib/releases/latest
    https://www.arduino.cc/reference/en/libraries/era
    https://registry.platformio.org/libraries/eoh-ltd/ERa/installation

    ERa website:                https://e-ra.io
    ERa blog:                   https://iotasia.org
    ERa forum:                  https://forum.eoh.io
    Follow us:                  https://www.fb.com/EoHPlatform
 *************************************************************/

// Enable debug console

#define ERA_DEBUG

/* Define MQTT host */
#define DEFAULT_MQTT_HOST "mqtt1.eoh.io"

// You should get Auth Token in the ERa App or ERa Dashboard
#define ERA_AUTH_TOKEN "6c46f012-8e42-41cd-a56c-c4edb45c0c53"

/* Define setting button */
// #define BUTTON_PIN              0

#if defined(BUTTON_PIN)
    // Active low (false), Active high (true)
    #define BUTTON_INVERT       false
    #define BUTTON_HOLD_TIMEOUT 5000UL

    // This directive is used to specify whether the configuration should be erased.
    // If it's set to true, the configuration will be erased.
    #define ERA_ERASE_CONFIG    false
#endif

#include <Arduino.h>
#include <ERa.hpp>
#include <DHT.h>
#define DHTTYPE DHT11   // DHT 22  (AM2302), AM2321
#define DHTPIN  1    // Digital pin connected to the DHT sensor
DHT dht(DHTPIN, DHTTYPE);
float temperature=0;
float humidity=0;
const char ssid[] = "NAH"; 
const char pass[] = "27122009";

WiFiClient mbTcpClient;

#if defined(ERA_AUTOMATION)
    #include <Automation/ERaSmart.hpp>

    #if defined(ESP32) || defined(ESP8266)
        #include <Time/ERaEspTime.hpp>
        ERaEspTime syncTime;
    #else
        #define USE_BASE_TIME

        #include <Time/ERaBaseTime.hpp>
        ERaBaseTime syncTime;
    #endif

    ERaSmart smart(ERa, syncTime);
#endif

#if defined(BUTTON_PIN)
    #include <ERa/ERaButton.hpp>

    ERaButton button;

    #if ERA_VERSION_NUMBER >= ERA_VERSION_VAL(1, 6, 0)
        static void eventButton(uint16_t pin, ButtonEventT event) {
            if (event != ButtonEventT::BUTTON_ON_HOLD) {
                return;
            }
            ERa.switchToConfig(ERA_ERASE_CONFIG);
            (void)pin;
        }
    #elif ERA_VERSION_NUMBER >= ERA_VERSION_VAL(1, 2, 0)
        static void eventButton(uint8_t pin, ButtonEventT event) {
            if (event != ButtonEventT::BUTTON_ON_HOLD) {
                return;
            }
            ERa.switchToConfig(ERA_ERASE_CONFIG);
            (void)pin;
        }
    #else
        static void eventButton(ButtonEventT event) {
            if (event != ButtonEventT::BUTTON_ON_HOLD) {
                return;
            }
            ERa.switchToConfig(ERA_ERASE_CONFIG);
        }
    #endif

    #if defined(ESP32)
        #include <pthread.h>

        pthread_t pthreadButton;

        static void* handlerButton(void* args) {
            for (;;) {
                button.run();
                ERaDelay(10);
            }
            pthread_exit(NULL);
        }

        void initButton() {
            pinMode(BUTTON_PIN, INPUT);
            button.setButton(BUTTON_PIN, digitalRead, eventButton,
                            BUTTON_INVERT).onHold(BUTTON_HOLD_TIMEOUT);
            pthread_create(&pthreadButton, NULL, handlerButton, NULL);
        }
    #elif defined(ESP8266)
        #include <Ticker.h>

        Ticker ticker;

        static void handlerButton() {
            button.run();
        }

        void initButton() {
            pinMode(BUTTON_PIN, INPUT);
            button.setButton(BUTTON_PIN, digitalRead, eventButton,
                            BUTTON_INVERT).onHold(BUTTON_HOLD_TIMEOUT);
            ticker.attach_ms(100, handlerButton);
        }
    #elif defined(ARDUINO_AMEBA)
        #include <GTimer.h>

        const uint32_t timerIdButton {0};

        static void handlerButton(uint32_t data) {
            button.run();
            (void)data;
        }

        void initButton() {
            pinMode(BUTTON_PIN, INPUT);
            button.setButton(BUTTON_PIN, digitalReadArduino, eventButton,
                            BUTTON_INVERT).onHold(BUTTON_HOLD_TIMEOUT);
            GTimer.begin(timerIdButton, (100 * 1000), handlerButton);
        }
    #endif
#endif

/* This function will run every time ERa is connected */
ERA_CONNECTED() {
    ERA_LOG(ERA_PSTR("ERa"), ERA_PSTR("ERa connected!"));
}

/* This function will run every time ERa is disconnected */
ERA_DISCONNECTED() {
    ERA_LOG(ERA_PSTR("ERa"), ERA_PSTR("ERa disconnected!"));
}

/* This function print uptime every second */
void timerEvent() {
     humidity=dht.readHumidity();
    temperature=dht.readTemperature();
    ERa.virtualWrite(V2,temperature);
    ERa.virtualWrite(V3,humidity); 
    ERA_LOG(ERA_PSTR("Timer"), ERA_PSTR("Uptime: %d"), ERaMillis() / 1000L);//ghi lai tg thiet bi da hoat dong tinh tu luc hoat dong(tg hien thi tren dashboard hoac serial monitor)

}

#if defined(USE_BASE_TIME)
    unsigned long getTimeCallback() {
        // Please implement your own function
        // to get the current time in seconds.
        return 0;
    } 
#endif

void setup() {
      dht.begin();
    /* Setup debug console */
#if defined(ERA_DEBUG)
    Serial.begin(115200);
#endif

#if defined(BUTTON_PIN)
    /* Initializing button. */
    initButton();
    /* Enable read/write WiFi credentials */
    ERa.setPersistent(true);
#endif

#if defined(USE_BASE_TIME)
    syncTime.setGetTimeCallback(getTimeCallback);
#endif

    /* Setup Client for Modbus TCP/IP */
    ERa.setModbusClient(mbTcpClient);

    /* Set scan WiFi. If activated, the board will scan
       and connect to the best quality WiFi. */
    ERa.setScanWiFi(true);
            
    /* Initializing the ERa library. */
    ERa.begin(ssid, pass);

    /* Setup timer called function every second */
    ERa.addInterval(1000L, timerEvent);//ham dc goi sau 1k mili giay
}

void loop() {
    ERa.run();
     float h = dht.readHumidity();
  // Read temperature as Celsius (the default)
  float t = dht.readTemperature();
}
