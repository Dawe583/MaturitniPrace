import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import numpy as np
import sys
import torch
import pandas as pd  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ultralytics import YOLO

class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message != '\n':  
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass  

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("600x400")  
        self.root.minsize(600, 400)  

        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  
        font_secondary = ("Helvetica", 6)  
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Helvetica", 12, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=10)

        
        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        
        self.image_frame = tk.Frame(self.image_console_frame, bg="#eeeeee", bd=1, relief="solid")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        
        self.canvas = tk.Canvas(self.image_frame, bg="black", height=250)  
        self.canvas.pack(fill="both", expand=True)

        
        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=6, bg="#222222", fg="#ffffff", font=("Courier", 10), bd=1, relief="solid")
        self.console_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.console_input = tk.Entry(self.console_frame, bg="#333333", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_input.pack(fill="x", padx=5, pady=5)
        self.console_input.bind("<Return>", self.execute_command)

        
        sys.stdout = OutputRedirector(self.console_output)

        self.redirect_console_output()

        
        self.controls_frame = tk.Frame(root, bg=bg_color, width=120) 
        self.controls_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

        
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_image_btn.pack(fill="x", pady=5)

        
        self.toggle_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.toggle_camera,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.toggle_camera_btn.pack(fill="x", pady=5)

        
        self.capture_image_btn = tk.Button(
            self.controls_frame, text="Vyfotit obrázek", command=self.capture_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.capture_image_btn.pack(fill="x", pady=5)

        
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        
        self.export_csv_btn = tk.Button(
            self.controls_frame, text="Exportovat do CSV", command=self.export_results_to_csv,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.export_csv_btn.pack(fill="x", pady=5)

        
        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        
        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        
        self.create_chart()
        self.update_chart([])  

        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  
        root.grid_columnconfigure(1, weight=2)  

       
        self.load_model() 

        
        self.camera_active = False  
        self.is_paused = False  

        
        self.detection_results = []

    def load_model(self):
        try:
            self.model = YOLO('/home/pi/maturitniprace/yolov8n.pt') 
            self.model.eval() 
            print("Model byl úspěšně načten.")
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodařilo se načíst model: {e}")

    def redirect_console_output(self):
        
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")

    def run_command(self, command):
       
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
        
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        
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
            self.start_camera()  
            self.toggle_camera_btn.config(text="Vypnout Kameru") 
        else:
            self.stop_camera()  
            self.toggle_camera_btn.config(text="Spustit Kameru") 

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_active = True  
        self.process_video()  

    def stop_camera(self):
        if hasattr(self, 'capture'):
            self.capture.release() 
            self.camera_active = False 
            self.canvas.delete("all")  

    def process_video(self):
        if hasattr(self, "capture") and not self.is_paused:
            ret, frame = self.capture.read()
            if ret:
                
                results = self.model(frame)

               
                if results:
                    
                    result = results[0]  
                    boxes = result.boxes  

                    for box in boxes:
                        
                        x1, y1, x2, y2 = box.xyxy[0]  
                        conf = box.conf[0]  
                        cls = box.cls[0]  

                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  
                        cv2.putText(frame, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

                        
                        print(f'Detected: {self.model.names[int(cls)]} with confidence {conf:.2f}\n')  
                        self.detection_results.append((self.model.names[int(cls)], conf, int(x1), int(y1), int(x2), int(y2)))  

                
                self.display_image(frame)

                
                self.update_chart([result[1] for result in self.detection_results])  

                
                self.root.after(10, self.process_video)

    def display_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image = image.resize((canvas_width, canvas_height), Image.LANCZOS)

        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def capture_image(self):
        if self.camera_active and hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                
                filename = "captured_image.png"
                cv2.imwrite(filename, frame)
                print(f"Image captured and saved as {filename}\n")
            else:
                print("Failed to capture image.\n")
        else:
            print("Camera is not active.\n")

    def detect_objects(self):
        
        if hasattr(self, "image"):
            results = self.model(self.image)
            if results:
                boxes = results[0].boxes 
                for box in boxes:
                    
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]

                    
                    cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) 
                    cv2.putText(self.image, f'{self.model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 

                    
                    print(f'Detected: {self.model.names[int(cls)]} with confidence {conf:.2f}\n')  
                    self.detection_results.append((self.model.names[int(cls)], conf, int(x1), int(y1), int(x2), int(y2)))  

                
                self.display_image(self.image)
                
                self.update_chart([result[1] for result in self.detection_results]) 
            else:
                print("No objects detected.")
        else:
            print("Please load an image first!")

    def edit_image(self):
        
        if hasattr(self, "image"):
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.image)
            self.image.show() 
        else:
            print("Please load an image first!")

    def create_chart(self):
       
        self.fig, self.ax = plt.subplots(figsize=(3, 2), dpi=80)
        self.ax.set_facecolor("#f0f0f0")
        self.ax.set_title("Confidence Scores", fontsize=8)
        self.ax.set_xlabel("Detection Index")
        self.ax.set_ylabel("Confidence Score")

        
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_chart.get_tk_widget().pack(fill="both", expand=True)

    def update_chart(self, data):
       
        self.ax.clear()
        self.ax.set_title("Confidence Scores", fontsize=8)
        self.ax.set_xlabel("Detection Index")
        self.ax.set_ylabel("Confidence Score")
        self.ax.plot(data, color='blue', linewidth=2)
        self.fig.canvas.draw()

    def export_results_to_csv(self):
        if self.detection_results:
            
            df = pd.DataFrame(self.detection_results, columns=["Object", "Confidence", "X1", "Y1", "X2", "Y2"])
            
            
            output_file = "detection_results.csv"
            
            
            df.to_csv(output_file, index=False)
            print(f"Results exported to {output_file}\n")
        else:
            print("No detection results to export.")

    def on_button_press(self, event):
        
        self.x1, self.y1 = event.x, event.y

    def on_mouse_drag(self, event):
        
        self.x2, self.y2 = event.x, event.y
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red", tags="rect")

    def on_button_release(self, event):
        
        self.x2, self.y2 = event.x, event.y
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red")


root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
