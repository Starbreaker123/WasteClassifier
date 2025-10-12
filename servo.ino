#include <WiFi.h>
#include <ESP32Servo.h>

// WiFi credentials
const char* ssid = "NAH";
const char* password = "27122009";

// Server IP address (Python computer IP)
// You need to replace this with your computer's actual IP address
const char* serverIP = "10.177.148.103";  // CHANGE THIS TO YOUR COMPUTER'S IP
const int serverPort = 8080;

// Hardware pins
#define CON1 7
#define CON2 15
#define SERVO_PIN 14

Servo myServo;
WiFiClient client;

// State variables
int state = 0; // 0: chờ CON1, 1: đang chạy camera, 2: chờ CON2, 3: xử lý kết quả
bool con1_pressed = false;
bool con2_pressed = false;
bool wifi_connected = false;
bool server_connected = false;

void setup() {
  pinMode(CON1, INPUT_PULLUP);
  pinMode(CON2, INPUT_PULLUP);
  
  Serial.begin(115200);
  
  // Initialize servo
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(90); // Neutral position
  
  Serial.println("ESP32 WiFi Waste Sorter Starting...");
  
  // Connect to WiFi
  connectToWiFi();
  
  // Connect to Python server
  connectToServer();
  
  Serial.println("ESP32 Waste Sorter Ready!");
  Serial.println("State 0: Waiting for CON1 trigger to start camera...");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifi_connected = true;
    Serial.println("");
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    wifi_connected = false;
    Serial.println("");
    Serial.println("WiFi connection failed!");
  }
}

void connectToServer() {
  if (!wifi_connected) {
    Serial.println("Cannot connect to server - WiFi not connected");
    return;
  }
  
  Serial.print("Connecting to Python server: ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);
  
  if (client.connect(serverIP, serverPort)) {
    server_connected = true;
    Serial.println("Connected to Python server!");
  } else {
    server_connected = false;
    Serial.println("Failed to connect to Python server!");
    Serial.println("Make sure Python script is running and IP address is correct");
  }
}

void sendCommand(String command) {
  if (server_connected && client.connected()) {
    client.println(command);
    Serial.print("Sent to server: ");
    Serial.println(command);
  } else {
    Serial.println("Cannot send command - server not connected");
    // Try to reconnect
    if (wifi_connected) {
      connectToServer();
    }
  }
}

int receiveResult() {
  if (server_connected && client.connected()) {
    // Wait for response with timeout
    unsigned long startTime = millis();
    while (!client.available() && (millis() - startTime < 10000)) {
      delay(100);
    }
    
    if (client.available()) {
      String response = client.readStringUntil('\n');
      response.trim();
      int result = response.toInt();
      Serial.print("Received from server: ");
      Serial.println(result);
      return result;
    } else {
      Serial.println("Timeout waiting for server response");
      return -1;
    }
  } else {
    Serial.println("Cannot receive result - server not connected");
    return -1;
  }
}

void checkWiFiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    wifi_connected = false;
    server_connected = false;
    Serial.println("WiFi connection lost - attempting reconnection...");
    connectToWiFi();
    if (wifi_connected) {
      connectToServer();
    }
  }
}

void loop() {
  // Check WiFi connection periodically
  static unsigned long lastWiFiCheck = 0;
  if (millis() - lastWiFiCheck > 30000) { // Check every 30 seconds
    checkWiFiConnection();
    lastWiFiCheck = millis();
  }
  
  switch (state) {
    case 0: // Chờ CON1 được kích hoạt
      if (digitalRead(CON1) == LOW && !con1_pressed) {
        con1_pressed = true;
        Serial.println("CON1 triggered - Starting camera detection!");
        
        // Gửi lệnh bật camera cho Python
        sendCommand("1");
        
        state = 1;
        Serial.println("State 1: Camera is running - Press CON2 to stop and get result");
        delay(300); // debounce
      }
      
      // Reset flag khi CON1 được thả
      if (digitalRead(CON1) == HIGH) {
        con1_pressed = false;
      }
      break;
      
    case 1: // Camera đang chạy - chờ CON2 để dừng và lấy kết quả
      if (digitalRead(CON2) == LOW && !con2_pressed) {
        con2_pressed = true;
        Serial.println("CON2 pressed - Stopping camera and requesting result...");
        
        // Gửi lệnh dừng camera và lấy kết quả cho Python
        sendCommand("2");
        
        state = 2;
        Serial.println("State 2: Waiting for result from server...");
        delay(300); // debounce
      }
      
      // Reset flag khi CON2 được thả
      if (digitalRead(CON2) == HIGH) {
        con2_pressed = false;
      }
      break;
      
    case 2: // Chờ kết quả từ server
      {
        int result = receiveResult();
        
        if (result != -1) { // Valid result received
          Serial.print("Received result from server: ");
          Serial.println(result);
          
          if (result == 1) {
            // Giá trị 0 - giữ nguyên góc (không quay)
            Serial.println("Result = 0: Keeping servo at neutral position (90°)");
            myServo.write(90);
            
          } else if (result == 0) {
            // Giá trị 1 - quay servo rồi quay lại vị trí ban đầu
            Serial.println("Result = 1: Moving servo to 0° then back to 90°");
            myServo.write(0);
            delay(5000); // Giữ vị trí 2 giây
            myServo.write(90);
            Serial.println("Servo returned to neutral position");
            
          } else {
            // Giá trị khác hoặc lỗi
            Serial.print("Error or invalid result: ");
            Serial.print(result);
            Serial.println(" - keeping servo at neutral");
            myServo.write(90);
          }
          
          // Chuyển sang state kết thúc
          state = 3;
        } else {
          // Timeout hoặc lỗi kết nối - thử lại hoặc chuyển sang state kết thúc
          Serial.println("Failed to receive result - ending cycle");
          myServo.write(90); // Keep neutral position
          state = 3;
        }
        
        delay(500);
      }
      break;
      
    case 3: // Kết thúc chu kỳ
      Serial.println("=== CYCLE COMPLETED ===");
      Serial.println("Ready for next cycle");
      
      // Reset về state 0
      state = 0;
      con1_pressed = false;
      con2_pressed = false;
      
      Serial.println("State 0: Waiting for CON1 trigger to start camera...");
      delay(1000);
      break;
  }
  
  // Check if server connection is still alive
  if (server_connected && !client.connected()) {
    Serial.println("Server connection lost");
    server_connected = false;
  }
  
  delay(50); // Delay nhỏ để tránh spam
}

void printWiFiStatus() {
  Serial.println("=== WiFi Status ===");
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Signal strength (RSSI): ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  Serial.print("Server connection: ");
  Serial.println(server_connected ? "Connected" : "Disconnected");
}