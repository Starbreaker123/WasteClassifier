const char MAIN_page[] PROGMEM = R"=====(
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      html {
        font-family: Arial;
        display: inline-block;
        margin: 0px auto;
        text-align: center;
      }

      h1 { font-size: 2.0rem; color:#2980b9;}
      
      .buttonON {
        display: inline-block;
        padding: 15px 25px;
        font-size: 24px;
        font-weight: bold;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #fff;
        background-color: #4CAF50;
        border: none;
        border-radius: 15px;
        box-shadow: 0 5px #999;
      }
      .buttonON:hover {background-color: #3e8e41}
      .buttonON:active {
        background-color: #3e8e41;
        box-shadow: 0 1px #666;
        transform: translateY(4px);
      }
.buttonOFF {
        display: inline-block;
        padding: 15px 25px;
        font-size: 24px;
        font-weight: bold;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        outline: none;
        color: #fff;
        background-color: #e74c3c;
        border: none;
        border-radius: 15px;
        box-shadow: 0 5px #999;
      }
      .buttonOFF:hover {background-color: #c0392b}
      .buttonOFF:active {
        background-color: #c0392b;
        box-shadow: 0 1px #666;
        transform: translateY(4px);
      }
      
      .textbox {
        font-size: 20px;
        padding: 10px;
        width: 60%;
        margin: 20px auto;
        border-radius: 5px;
        border: 1px solid #ccc;
      }
    </style>
  </head>
  
  <body>
    <div>
      <h1>Arduino IDE ESP32 Web Server Station Mode</h1>
      <h2>ESP32 Web Server Controlling LED</h2><br>
    </div>

    <div>
      <p style="color:#2c3e50;font-weight: bold;font-size: 24px;">LED (GPIO 18) Status is : <span id="LEDState1">NA</span></p>
    </div>

    <div>
      <button type="button" class="buttonON" onclick="sendDataLED1(0)">LED ON</button>
      <button type="button" class="buttonOFF" onclick="sendDataLED1(1)">LED OFF</button><BR>
    </div>

    <div>
      <p style="color:#2c3e50;font-weight: bold;font-size: 24px;">LED (GPIO 19) Status is : <span id="LEDState2">NA</span></p>
    </div>
    
    <div>
      <button type="button" class="buttonON" onclick="sendDataLED2(0)">LED ON</button>
      <button type="button" class="buttonOFF" onclick="sendDataLED2(1)">LED OFF</button><BR>
    </div>F

    <!-- Added Text Box Section -->
    <div>
      <input type="text" id="userInput" class="textbox" placeholder="Enter your message">
      <button type="button" class="buttonON" onclick="submitText()">Submit</button>
      <p id="userMessage"></p>
    </div>

    <script>
      function sendDataLED1(LED) {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
            document.getElementById("LEDState1").innerHTML = this.responseText;
          }  
        };
        xhttp.open("GET", "setLED1?LEDState="+LED, true);
        xhttp.send(); 
      }
      
      function sendDataLED2(LED) {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
            document.getElementById("LEDState2").innerHTML = this.responseText;
          }  
        };
        xhttp.open("GET", "setLED2?LEDState="+LED, true);
        xhttp.send(); 
      }

      // Function to handle the submission of text
     function submitText() {
  var userInput = document.getElementById("userInput").value; // Lấy giá trị từ text box
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      document.getElementById("userMessage").innerHTML = this.responseText;
    }
  };

  // Gửi yêu cầu với tham số userInput từ text box
  xhttp.open("GET", "/submitText?userMessage=" + encodeURIComponent(userInput), true);
  xhttp.send();
}

    </script>
  </body>
</html>
)=====";