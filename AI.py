import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import time
import threading
import socket
import json
from collections import deque, Counter


class WiFiCommunicator:
    """Handles WiFi communication with ESP32."""

    def __init__(self, server_port=8080):
        self.server_port = server_port
        self.server_socket = None
        self.esp32_socket = None
        self.is_server_running = False
        self.server_thread = None
        self.last_command = None

    def get_local_ip(self):
        """Get the local IP address."""
        try:
            # Connect to a remote address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "192.168.1.100"  # Default fallback IP

    def start_server(self):
        """Start TCP server to communicate with ESP32."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            local_ip = self.get_local_ip()
            self.server_socket.bind((local_ip, self.server_port))
            self.server_socket.listen(1)

            self.is_server_running = True
            print(f"WiFi Server started on {local_ip}:{self.server_port}")
            print("Waiting for ESP32 connection...")
            print(f"Configure ESP32 to connect to: {local_ip}:{self.server_port}")

            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            return True

        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def _server_loop(self):
        """Server loop to handle ESP32 connections."""
        while self.is_server_running:
            try:
                esp32_socket, addr = self.server_socket.accept()
                print(f"ESP32 connected from: {addr}")
                self.esp32_socket = esp32_socket

                # Handle ESP32 communication
                self._handle_esp32_communication()

            except Exception as e:
                if self.is_server_running:
                    print(f"Server error: {e}")
                break

    def _handle_esp32_communication(self):
        """Handle incoming messages from ESP32."""
        buffer = ""
        while self.is_server_running and self.esp32_socket:
            try:
                data = self.esp32_socket.recv(1024).decode('utf-8')
                if not data:
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line:
                        print(f"Received from ESP32: {line}")
                        self.last_command = line

            except Exception as e:
                print(f"ESP32 communication error: {e}")
                break

        # Connection lost
        print("ESP32 disconnected")
        self.esp32_socket = None

    def send_result_to_esp32(self, result):
        """Send detection result to ESP32."""
        if self.esp32_socket:
            try:
                message = f"{result}\n"
                self.esp32_socket.send(message.encode('utf-8'))
                print(f"Sent result to ESP32: {result}")
                return True
            except Exception as e:
                print(f"Failed to send result to ESP32: {e}")
                self.esp32_socket = None
                return False
        return False

    def read_command(self):
        """Get the last command from ESP32 and clear it."""
        cmd = self.last_command
        self.last_command = None
        return cmd

    def send_result(self, result):
        """Send result to ESP32 (alias for compatibility)."""
        return self.send_result_to_esp32(result)

    def stop_server(self):
        """Stop the server."""
        self.is_server_running = False
        if self.esp32_socket:
            self.esp32_socket.close()
        if self.server_socket:
            self.server_socket.close()


class WasteDetector:
    def __init__(self, model_path):
        """Initialize the waste detector with the trained model."""
        self.model_path = model_path
        self.model = None
        self.input_shape = None
        self.load_model()

    def load_model(self):
        """Load the Keras model from h5 file."""
        try:
            self.model = keras.models.load_model(self.model_path)
            self.input_shape = self.model.input_shape[1:3]  # Get height, width
            print(f"Model loaded successfully!")
            print(f"Expected input shape: {self.input_shape}")
            print(f"Model input shape: {self.model.input_shape}")

            # Warm up the model with a dummy prediction
            dummy_input = np.random.random((1, *self.input_shape, 3)).astype(np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("Model warmed up!")

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True

    def enhance_image_lighting(self, image):
        """Enhance image lighting and contrast for better detection."""
        # Convert to LAB color space for better lighting adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Additional brightness and contrast adjustment if needed
        alpha = 1.1  # Contrast control (1.0-3.0)
        beta = 10  # Brightness control (0-100)
        enhanced_bgr = cv2.convertScaleAbs(enhanced_bgr, alpha=alpha, beta=beta)

        return enhanced_bgr

    def preprocess_image(self, image):
        """Preprocess the image for model prediction."""
        # Enhance lighting first
        enhanced_image = self.enhance_image_lighting(image)

        # Resize image to model's expected input size
        resized = cv2.resize(enhanced_image, self.input_shape)

        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Always keep RGB - do not convert to grayscale
        processed_image = rgb_image

        # Normalize pixel values to 0-1 range
        normalized = processed_image.astype(np.float32) / 255.0

        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)

        return batch_image

    def predict(self, image):
        """Make prediction on the input image."""
        if self.model is None:
            return None, 0.0

        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)

            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)

            # Get the prediction value and confidence
            pred_value = prediction[0][0]  # Assuming single output

            # Convert to binary classification (0 or 1)
            binary_pred = 1 if pred_value > 0.5 else 0
            confidence = pred_value if binary_pred == 1 else (1 - pred_value)

            return binary_pred, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def get_label(self, prediction):
        """Convert prediction to human readable label."""
        if prediction == 1:
            return "ORGANIC"
        elif prediction == 0:
            return "INORGANIC"
        else:
            return "ERROR"


class WiFiIntegratedWasteDetectionCamera:
    def __init__(self, detector, wifi_communicator, camera_index=0):
        self.detector = detector
        self.wifi_comm = wifi_communicator
        self.camera_index = camera_index

        # System states
        self.system_running = False
        self.camera_active = False

        # Threading variables
        self.frame_queue = deque(maxlen=2)
        self.prediction_results = []  # Store all predictions during camera session

        # Camera variables
        self.cap = None
        self.running = False

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Detection timing
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # 5 FPS detection

    def initialize_camera(self):
        """Initialize camera with optimized settings."""
        print(f"Initializing camera {self.camera_index}...")

        # Use DirectShow backend for better performance on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index} with DirectShow, trying default...")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Set resolution for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass

        # Test capture
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("Error: Could not capture test frame")
            return False

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {actual_width}x{actual_height}")

        return True

    def close_camera(self):
        """Close camera and release resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print("Camera closed")

    def capture_and_process_frames(self):
        """Continuously capture frames and make predictions while camera is active."""
        consecutive_failures = 0
        max_failures = 10

        while self.running and self.camera_active:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    frame_mean = np.mean(frame)
                    if frame_mean > 5:  # Not completely black
                        self.frame_queue.append(frame)
                        consecutive_failures = 0

                        # Update FPS counter
                        self.fps_counter += 1
                        current_time = time.time()
                        if current_time - self.fps_start_time >= 1.0:
                            self.current_fps = self.fps_counter
                            self.fps_counter = 0
                            self.fps_start_time = current_time

                        # Make predictions at regular intervals
                        if current_time - self.last_prediction_time >= self.prediction_interval:
                            prediction, confidence = self.detector.predict(frame)

                            if prediction is not None:
                                result = {
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'label': self.detector.get_label(prediction),
                                    'timestamp': current_time
                                }
                                self.prediction_results.append(result)
                                print(f"Detection: {result['label']} ({confidence:.2%})")

                            self.last_prediction_time = current_time

                        # Display preview
                        self.display_frame(frame)

                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1

            if consecutive_failures >= max_failures:
                print("Too many capture failures, stopping camera...")
                break

            time.sleep(0.001)

    def display_frame(self, frame):
        """Display current frame with overlay information."""
        display_frame = frame.copy()

        # Add status overlay
        status = "CAMERA ACTIVE - Waiting for CON2..."
        cv2.putText(display_frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {self.current_fps}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Predictions: {len(self.prediction_results)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show last prediction
        if self.prediction_results:
            last_result = self.prediction_results[-1]
            color = (0, 255, 0) if last_result['prediction'] == 1 else (0, 0, 255)
            text = f"Last: {last_result['label']} ({last_result['confidence']:.2%})"
            cv2.putText(display_frame, text,
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # WiFi status
        wifi_status = "WiFi: Connected" if self.wifi_comm.esp32_socket else "WiFi: Waiting for ESP32"
        wifi_color = (0, 255, 0) if self.wifi_comm.esp32_socket else (0, 255, 255)
        cv2.putText(display_frame, wifi_status,
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wifi_color, 2)

        cv2.imshow('WiFi Waste Detection Camera', display_frame)
        cv2.waitKey(1)

    def start_camera(self):
        """Start camera and begin detection."""
        print("Starting camera...")

        # Initialize camera
        if not self.initialize_camera():
            return False

        # Reset prediction results
        self.prediction_results = []
        self.camera_active = True
        self.running = True
        self.last_prediction_time = 0

        # Start capture and processing thread
        self.capture_thread = threading.Thread(target=self.capture_and_process_frames, daemon=True)
        self.capture_thread.start()

        print("Camera started successfully")
        return True

    def stop_camera_and_get_result(self):
        """Stop camera and analyze all predictions to return final result."""
        print("Stopping camera and analyzing results...")

        # Stop camera
        self.camera_active = False
        self.running = False

        # Wait for capture thread to finish
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)

        # Close camera
        self.close_camera()
        cv2.destroyAllWindows()

        # Analyze results
        return self.analyze_predictions()

    def analyze_predictions(self):
        """Analyze all predictions and determine final result."""
        if not self.prediction_results:
            print("No predictions made during camera session")
            return 0  # Default to inorganic if no predictions

        # Count predictions
        predictions = [result['prediction'] for result in self.prediction_results]
        prediction_counter = Counter(predictions)

        organic_count = prediction_counter.get(1, 0)
        inorganic_count = prediction_counter.get(0, 0)
        total_predictions = len(predictions)

        print(f"\nDetection Results Summary:")
        print(f"Total predictions: {total_predictions}")
        print(f"Organic predictions: {organic_count}")
        print(f"Inorganic predictions: {inorganic_count}")

        # Determine final result (1 for organic majority, 0 for inorganic majority or tie)
        if organic_count > inorganic_count:
            final_result = 1
            final_label = "ORGANIC"
        else:
            final_result = 0
            final_label = "INORGANIC"

        confidence_percentage = max(organic_count, inorganic_count) / total_predictions * 100

        print(f"Final Result: {final_label} ({confidence_percentage:.1f}% confidence)")
        print(f"Sending result {final_result} to ESP32")
        return final_result

    def listen_for_esp_commands(self):
        """Listen for commands from ESP32 via WiFi."""
        print("Listening for ESP32 WiFi commands...")
        print("Commands: '1' = Start Camera, '2' = Stop Camera and Get Result")

        while self.system_running:
            command = self.wifi_comm.read_command()

            if command == "1":
                print("Received START CAMERA command from ESP32")

                # Start camera if not already running
                if not self.camera_active:
                    success = self.start_camera()
                    if not success:
                        print("Failed to start camera, sending error to ESP32")
                        self.wifi_comm.send_result(-1)
                else:
                    print("Camera already active")

            elif command == "2":
                print("Received STOP CAMERA command from ESP32")

                if self.camera_active:
                    # Stop camera and get result
                    result = self.stop_camera_and_get_result()

                    # Send result back to ESP32
                    self.wifi_comm.send_result(result)
                    print(f"Sent result {result} to ESP32")
                else:
                    print("Camera not active, sending default result 0")
                    self.wifi_comm.send_result(0)

            elif command and command.isdigit():
                # Handle any other numeric commands
                print(f"Received unknown command: {command}")

            time.sleep(0.1)  # Small delay to prevent busy waiting

    def run_system(self):
        """Run the complete WiFi integrated system."""
        print("Starting WiFi Integrated Waste Detection System...")
        print("Logic: CON1 starts camera → CON2 stops camera and gets result")

        # Start WiFi server and wait for ESP32 connection
        if not self.wifi_comm.start_server():
            print("Failed to start WiFi server. Exiting...")
            return

        self.system_running = True

        try:
            # Listen for ESP32 commands
            self.listen_for_esp_commands()

        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        except Exception as e:
            print(f"System error: {e}")
        finally:
            # Cleanup
            self.system_running = False
            self.camera_active = False
            self.running = False
            self.close_camera()
            cv2.destroyAllWindows()
            self.wifi_comm.stop_server()
            print("System shutdown complete")


def main():
    # Initialize the waste detector
    model_path = r"C:\Users\ADMIN\Documents\Arduino\AI\waste_classifier_model.h5"
    detector = WasteDetector(model_path)

    if detector.model is None:
        print("Failed to load model. Exiting...")
        return

    # Initialize WiFi communicator
    wifi_comm = WiFiCommunicator(server_port=8080)

    # Initialize integrated WiFi camera system
    camera_system = WiFiIntegratedWasteDetectionCamera(
        detector=detector,
        wifi_communicator=wifi_comm,
        camera_index=0  # Default camera
    )

    # Run the system
    camera_system.run_system()


if __name__ == "__main__":
    main()
