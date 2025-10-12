#include <WiFi.h>
#include <WebServer.h>

// WiFi credentials
const char* ssid = "NAH";
const char* password = "27122009";

// Tạo server trên cổng 80
WebServer server(80);

// GPIO
const int LED_1 = 7;   // Nếu nhận data = 1
const int LED_2 = 41;  // Nếu nhận data = 2

void setup() {
  Serial.begin(115200);  // Bật Serial Monitor để in thông tin

  WiFi.begin(ssid, password);
  Serial.print("Đang kết nối WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // ✅ In ra khi đã kết nối WiFi thành công
  Serial.println("\n✅ Đã kết nối WiFi!");
  Serial.print("🖥️ IP của ESP32: ");
  Serial.println(WiFi.localIP());

  pinMode(LED_1, OUTPUT);
  pinMode(LED_2, OUTPUT);
  digitalWrite(LED_1, LOW);
  digitalWrite(LED_2, LOW);

  server.on("/value", HTTP_GET, handleValue);
  server.begin();
}


void handleValue() {
  if (server.hasArg("data")) {
    int data = server.arg("data").toInt();

    if (data == 1) {
      digitalWrite(LED_1, HIGH);
      digitalWrite(LED_2, LOW);
    } else if (data == 2) {
      digitalWrite(LED_1, LOW);
      digitalWrite(LED_2, HIGH);
    }

    server.send(200, "text/plain", "OK");
  } else {
    // Không có data → không phản hồi gì cụ thể, không in
    server.send(204, "text/plain", "");
  }
}

void loop() {
  server.handleClient();
}
