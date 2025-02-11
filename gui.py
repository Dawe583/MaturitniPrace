import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import sys
import numpy as np
import tensorflow as tf

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
        self.root.geometry("700x400")  # Velikost okna
        self.root.minsize(700, 400)  # Minimální velikost okna

        # Barevné schéma
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  # Zmenšení písma pro ovládací prvky
        font_secondary = ("Helvetica", 6)  # Menší písmo pro textové komponenty
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
        self.controls_frame = tk.Frame(root, bg=bg_color, width=120)  # Zúžený ovládací panel
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

        # Inicializace TensorFlow Lite modelu pro detekci objektů
        self.interpreter = tf.lite.Interpreter(model_path="efficientdet_lite0.tflite")
        self.interpreter.allocate_tensors()

        # Rámec pro graf pod tlačítky
        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        self.rect = None  # Inicializace proměnné pro výběr oblasti
        self.create_chart()

        # Přidání události pro výběr oblasti
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Nastavení váhy sloupců
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 pro obrazovku
        root.grid_columnconfigure(1, weight=2)  # 2/5 pro ovládací panel

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulace příkazu pro zobrazení výstupu

    def run_command(self, command):
        # Spustí příkaz v PowerShellu a vypisuje jeho výstup do konzolového widgetu
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
        # Vložení výstupu do konzolového widgetu
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        # Získá příkaz od uživatele a spustí ho v PowerShellu
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
        self.cap = cv2.VideoCapture(2)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
            self.detect_objects(frame)  # Detekce objektů na každém snímku
            self.root.after(10, self.show_frame)

    def show_frame_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.display_image(frame)
            self.detect_objects(frame)  # Detekce objektů na každém snímku
            self.root.after(10, self.show_frame_video)
        else:
            self.video_capture.release()

    def display_image(self, img):
        self.original_image = img.copy()  # Uložení originálního obrázku pro úpravy
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((canvas_width, canvas_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

    def create_chart(self):
        # Vytvoření základního grafu
        self.figure, self.ax = plt.subplots(figsize=(4, 2))  # Menší graf
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvas.draw()

    def update_chart(self, detections):
        # Aktualizace dat v grafu
        self.ax.clear()
        if detections:
            labels = [d['object'] for d in detections]
            values = [d['confidence'] * 100 for d in detections]
            self.ax.bar(labels, values, color='blue')
            self.ax.set_ylim(0, 100)
            self.ax.set_title('Detekce objektů')
        self.chart_canvas.draw()

    def detect_objects(self, frame=None):
        # Provedení detekce objektů na obrázku nebo video snímku
        if frame is None:
            return
        
        # Předzpracování snímku
        input_tensor = self.process_image(frame)

        # Provádění inference
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Získání výsledků
        boxes = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]  # kořeny ohraničujících boxů
        class_ids = self.interpreter.get_tensor(self.interpreter.get_output_details()[1]['index'])[0]  # ID tříd
        scores = self.interpreter.get_tensor(self.interpreter.get_output_details()[2]['index'])[0]  # skóre detekcí

        # Filtrace detekcí na základě prahu skóre
        detections = []
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Threshold pro detekci
                detections.append({
                    'object': str(class_ids[i]),  # ID třídy
                    'confidence': scores[i],
                    'box': boxes[i]  # Koordináty boxu
                })

        # Aktualizace grafu a zobrazení detekcí na canvasu
        self.update_chart(detections)
        self.display_detections(frame, detections)

    def process_image(self, frame):
        # Předzpracování snímku pro TensorFlow Lite model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))  # Změna velikosti podle modelu
        input_tensor = np.expand_dims(img_resized, axis=0)
        input_tensor = np.asarray(input_tensor, dtype=np.float32)
        return input_tensor

    def display_detections(self, frame, detections):
        # Zobrazení detekcí na obrazovce
        for detection in detections:
            ymin, xmin, ymax, xmax = detection['box']
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

        # Převod zpět do formátu pro Tkinter
        self.display_image(frame)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="blue", width=2, tags="selection")

    def on_button_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="blue", width=2)
        print(f"Výběr oblasti: ({self.start_x}, {self.start_y}) -> ({self.end_x}, {self.end_y})")

    def close_application(self):
        if hasattr(self, "cap"):
            self.cap.release()
        if hasattr(self, "video_capture"):
            self.video_capture.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_application)
    root.mainloop()
