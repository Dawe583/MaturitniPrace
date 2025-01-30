def detect_objects(self):
    if hasattr(self, 'original_image'):
        # Save the current image to a temporary file
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, self.original_image)

        # Specify the model path
        model_path = "/home/pi/examples/lite/examples/object_detection/raspberry_pi/efficientdet_lite0.tflite"

        # Command to run the object detection script
        command = f"python3 object_detection.py {temp_image_path} {model_path}"

        # Run the command in a separate thread
        def execute_detection():
            try:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in iter(process.stdout.readline, ""):
                    self.insert_to_console(line)  # Insert output into console
                for line in iter(process.stderr.readline, ""):
                    self.insert_to_console(line, error=True)  # Insert errors into console
                process.stdout.close()
                process.stderr.close()
                process.wait()
            except Exception as e:
                self.insert_to_console(f"[ERROR] {str(e)}", error=True)

        # Start detection in a new thread
        thread = threading.Thread(target=execute_detection)
        thread.start()
    else:
        self.insert_to_console("[ERROR] No image loaded for detection.", error=True)