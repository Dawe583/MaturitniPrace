import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message != '\n':  # Ignorovat prázdné nové řádky
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass  # Není potřeba implementovat pro Tkinter, stačí ignorovat

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("700x400")
        self.root.minsize(700, 400)

        # Barevné schéma
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)
        font_secondary = ("Helvetica", 6)
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        # Nadpis aplikace
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Helvetica", 12, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=10)

        # Rámec pro zobrazení obrázku a konzoli
        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Rámec pro zobrazení obrázku
        self.image_frame = tk.Frame(self.image_console_frame, bg="#eeeeee", bd=1, relief="solid")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Konzolový výstup a vstup
        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=4, bg="#222222", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.console_input = tk.Entry(self.console_frame, bg="#333333", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_input.pack(fill="x", padx=5, pady=5)
        self.console_input.bind("<Return>", self.execute_command)

        self.redirect_console_output()

        # Ovládací panel a graf
        self.controls_frame = tk.Frame(root, bg=bg_color, width=120)
        self.controls_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

        # Tlačítka pro načítání obrázku a videa
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_image_btn.pack(fill="x", pady=5)

        self.load_video_btn = tk.Button(
            self.controls_frame, text="Načíst video", command=self.load_video,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_video_btn.pack(fill="x", pady=5)

        self.start_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.start_camera,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.start_camera_btn.pack(fill="x", pady=5)

        # Tlačítko pro rozpoznání objektů
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        # Možnost úprav obrázků
        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        # Přidání možnosti nastavení prahu pro detekci
        self.confidence_threshold = tk.Scale(
            self.controls_frame, from_=0, to=100, orient="horizontal", label="Prahová hodnota detekce (%)",
            bg=bg_color, fg="black", font=font_secondary
        )
        self.confidence_threshold.set(50)
        self.confidence_threshold.pack(fill="x", pady=5)

        # Inicializace YOLO modelu
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov8n.pt")
        
        # Nastavení váhy sloupců
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 pro obrazovku
        root.grid_columnconfigure(1, weight=2)  # 2/5 pro ovládací panel

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")

    def run_command(self, command):
        def execute():
            process = subprocess.Popen(["powershell", "-Command", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.show_frame_video()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.detect_and_display(frame)
            self.root.after(10, self.show_frame)

    def show_frame_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.detect_and_display(frame)
            self.root.after(10, self.show_frame_video)
        else:
            self.video_capture.release()

    def detect_and_display(self, frame):
        # Detekce objektů
        results = self.model(frame)
        detections = results.xywh[0].numpy()  # Detekce ve formátu [x, y, width, height, confidence]
        
        # Prahování
        threshold = self.confidence_threshold.get() / 100.0
        for detection in detections:
            confidence = detection[4]
            if confidence >= threshold:
                x, y, w, h = detection[:4]
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{results.names[int(detection[5])]} {confidence*100:.2f}%", 
                            (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.display_image(frame)

    def display_image(self, img):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((canvas_width, canvas_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

    def edit_image(self):
        if hasattr(self, 'image'):
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
