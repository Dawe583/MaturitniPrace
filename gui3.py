import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import numpy as np
import sys
import torch
import pandas as pd  # Import pandas for CSV export
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ultralytics import YOLO

class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message != '\n':  # Ignore empty new lines
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass  # No need to implement for Tkinter, just ignore

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("600x400")  # Set the window size
        self.root.minsize(600, 400)  # Minimum window size

        # Color scheme
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  # Smaller font for controls
        font_secondary = ("Helvetica", 6)  # Smaller font for text components
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        # Application title
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Helvetica", 12, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=10)

        # Frame for image display and console
        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Frame for image display
        self.image_frame = tk.Frame(self.image_console_frame, bg="#eeeeee", bd=1, relief="solid")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Adjust the height of the canvas to make room for the chart
        self.canvas = tk.Canvas(self.image_frame, bg="black", height=250)  # Set a fixed height for the canvas
        self.canvas.pack(fill="both", expand=True)

        # Console output and input
        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=6, bg="#222222", fg="#ffffff", font=("Courier", 10), bd=1, relief="solid")
        self.console_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.console_input = tk.Entry(self.console_frame, bg="#333333", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_input.pack(fill="x", padx=5, pady=5)
        self.console_input.bind("<Return>", self.execute_command)

        # Redirect console output to the text widget
        sys.stdout = OutputRedirector(self.console_output)

        self.redirect_console_output()

        # Control panel and chart
        self.controls_frame = tk.Frame(root, bg=bg_color, width=120)  # Narrow control panel
        self.controls_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

        # Buttons for loading image
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_image_btn.pack(fill="x", pady=5)

        # Button to toggle camera
        self.toggle_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.toggle_camera,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.toggle_camera_btn.pack(fill="x", pady=5)

        # Button to capture image
        self.capture_image_btn = tk.Button(
            self.controls_frame, text="Vyfotit obrázek", command=self.capture_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.capture_image_btn.pack(fill="x", pady=5)

        # Button for object detection
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        # Button to export results to CSV
        self.export_csv_btn = tk.Button(
            self.controls_frame, text="Exportovat do CSV", command=self.export_results_to_csv,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.export_csv_btn.pack(fill="x", pady=5)

        # Image editing option
        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        # Frame for chart under buttons
        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        # Initialize chart
        self.create_chart()
        self.update_chart([])  # Initialize empty chart

        # Add event for area selection
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Set row and column weights
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 for screen
        root.grid_columnconfigure(1, weight=2)  # 2/5 for control panel

        # Initialize YOLO model
        self.load_model()  # Load model

        # Camera state management
        self.camera_active = False  # Flag to track camera state
        self.is_paused = False  # Flag to track pause state

        # List to store detection results for CSV export
        self.detection_results = []

    def load_model(self):
        try:
            self.model = YOLO('/home/pi/maturitniprace/yolov8n.pt')  # Adjust as needed
            self.model.eval()  # Set model to evaluation mode
            print("Model byl úspěšně načten.")
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodařilo se načíst model: {e}")

    def redirect_console_output(self):
        # Redirect console output to Text widget
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulate command to show output

    def run_command(self, command):
        # Execute command in bash and print its output to console widget
        def execute():
            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in iter(process.stdout.readline, ""):
                self.insert_to_console(line)
            for line in iter(process.stderr.readline, ""):
                self.insert_to_console(line, error=True)
            process.stdout.close()
            process.stderr.close()
            process.wait()

        thread = threading.Thread(target=execute)
        thread.start()

    def insert_to_console(self, line, error=False):
        # Insert output into console widget
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        # Get command from user and execute it in bash
        command = self.console_input.get()
        if command.strip():
            self.console_output.insert(tk.END, f"> {command}\n")
            self.console_input.delete(0, tk.END)
            self.run_command(command)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()  # Start the camera
            self.toggle_camera_btn.config(text="Vypnout Kameru")  # Update button text
        else:
            self.stop_camera()  # Stop the camera
            self.toggle_camera_btn.config(text="Spustit Kameru")  # Update button text

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_active = True  # Set camera state to active
        self.process_video()  # Start processing video frames

    def stop_camera(self):
        if hasattr(self, 'capture'):
            self.capture.release()  # Release the camera
            self.camera_active = False  # Set camera state to inactive
            self.canvas.delete("all")  # Clear the canvas if needed

    def process_video(self):
        if hasattr(self, "capture") and not self.is_paused:
            ret, frame = self.capture.read()
            if ret:
                # Object detection on the frame
                results = self.model(frame)

                # Check if results are available
                if results:
                    # Extract the first result
                    result = results[0]  # Get the first result (for the first image)
                    boxes = result.boxes  # Access the boxes attribute

                    for box in boxes:
                        # Unpack the box
                        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates
                        conf = box.conf[0]  # Get the confidence score
                        cls = box.cls[0]  # Get the class index

                        # Draw the bounding box on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw rectangle
                        cv2.putText(frame, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

                        # Print detection results to the console
                        print(f'Detected: {self.model.names[int(cls)]} with confidence {conf:.2f}\n')  # New line added
                        self.detection_results.append((self.model.names[int(cls)], conf, int(x1), int(y1), int(x2), int(y2)))  # Store results for CSV

                # Display the result on the canvas
                self.display_image(frame)

                # Update the graph with the latest confidence scores
                self.update_chart([result[1] for result in self.detection_results])  # Update with confidence scores

                # Schedule the next frame processing
                self.root.after(10, self.process_video)

    def display_image(self, image):
        # Display image on canvas
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image = image.resize((canvas_width, canvas_height), Image.LANCZOS)  # Resize using LANCZOS

        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def capture_image(self):
        if self.camera_active and hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                # Save the captured frame as an image
                filename = "captured_image.png"
                cv2.imwrite(filename, frame)
                print(f"Image captured and saved as {filename}\n")
            else:
                print("Failed to capture image.\n")
        else:
            print("Camera is not active.\n")

    def detect_objects(self):
        # Recognize objects in the image
        if hasattr(self, "image"):
            results = self.model(self.image)
            if results:
                boxes = results[0].boxes  # Get the first result (for the first image)
                for box in boxes:
                    # Unpack the box
                    x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates
                    conf = box.conf[0]  # Get the confidence score
                    cls = box.cls[0]  # Get the class index

                    # Draw the bounding box on the image
                    cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw rectangle
                    cv2.putText(self.image, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

                    # Print detection results to the console
                    print(f'Detected: {self.model.names[int(cls)]} with confidence {conf:.2f}\n')  # New line added
                    self.detection_results.append((self.model.names[int(cls)], conf, int(x1), int(y1), int(x2), int(y2)))  # Store results for CSV

                # Display the detected image
                self.display_image(self.image)
                # Update the graph with the latest confidence scores
                self.update_chart([result[1] for result in self.detection_results])  # Update with confidence scores
            else:
                print("No objects detected.")
        else:
            print("Please load an image first!")

    def edit_image(self):
        # Function for image editing
        if hasattr(self, "image"):
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.image)
            self.image.show()  # Display edited image
        else:
            print("Please load an image first!")

    def create_chart(self):
        # Create chart for control panel
        self.fig, self.ax = plt.subplots(figsize=(3, 2), dpi=80)
        self.ax.set_facecolor("#f0f0f0")
        self.ax.set_title("Confidence Scores", fontsize=8)
        self.ax.set_xlabel("Detection Index")
        self.ax.set_ylabel("Confidence Score")

        # Create a canvas for the chart
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_chart.get_tk_widget().pack(fill="both", expand=True)

    def update_chart(self, data):
        # Update chart with confidence scores
        self.ax.clear()
        self.ax.set_title("Confidence Scores", fontsize=8)
        self.ax.set_xlabel("Detection Index")
        self.ax.set_ylabel("Confidence Score")
        self.ax.plot(data, color='blue', linewidth=2)
        self.fig.canvas.draw()

    def export_results_to_csv(self):
        if self.detection_results:
            # Create a DataFrame from the results
            df = pd.DataFrame(self.detection_results, columns=["Object", "Confidence", "X1", "Y1", "X2", "Y2"])
            
            # Specify the output CSV file name
            output_file = "detection_results.csv"
            
            # Export the DataFrame to a CSV file
            df.to_csv(output_file, index=False)
            print(f"Results exported to {output_file}\n")  # Notify user in console
        else:
            print("No detection results to export.")

    def on_button_press(self, event):
        # Capture button press for area selection
        self.x1, self.y1 = event.x, event.y

    def on_mouse_drag(self, event):
        # Capture mouse drag for area selection
        self.x2, self.y2 = event.x, event.y
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red", tags="rect")

    def on_button_release(self, event):
        # Capture button release to complete area selection
        self.x2, self.y2 = event.x, event.y
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red")

# Create main application window
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()