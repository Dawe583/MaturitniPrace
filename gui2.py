import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import numpy as np
import sys
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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

        # Rámec pro graf pod tlačítky
        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        # Inicializace grafu
        self.create_chart()
        self.update_chart([])  # Inicializace prázdného grafu

        # Přidání události pro výběr oblasti
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Nastavení váhy sloupců
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 pro obrazovku
        root.grid_columnconfigure(1, weight=2)  # 2/5 pro ovládací panel

        # Inicializace YOLO modelu
        self.load_model()  # Načtení modelu

    def load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/pi/maturitniprace/yolov8n.pt')  # Upravte podle potřeby
            self.model.eval()  # Nastavení modelu do evaluačního režimu
            print("Model byl úspěšně načten.")
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodařilo se načíst model: {e}")

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulace příkazu pro zobrazení výstupu

    def run_command(self, command):
        # Spustí příkaz v bash a vypisuje jeho výstup do konzolového widgetu
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
        # Vložení výstupu do konzolového widgetu
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        # Získá příkaz od uživatele a spustí ho v bash
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
            self.video = cv2.VideoCapture(file_path)
            self.process_video()

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.process_video()

    def process_video(self):
        # Zpracování videa a detekce objektů
        if hasattr(self, "capture"):
            ret, frame = self.capture.read()
            if ret:
                # Detekce objektů na snímku
                results = self.model(frame)
                # Získání prvního rámce s detekovanými objekty
                detected_image = results.render()[0]  # Získání prvního rámce s detekovanými objekty

                # Zobrazení výsledku na plátno
                self.display_image(detected_image)

                self.root.after(10, self.process_video)

    def display_image(self, image):
        # Zobrazení obrázku na plátno
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Změna velikosti obrázku na velikost plátna
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image = image.resize((canvas_width, canvas_height), Image.LANCZOS)  # Změna na LANCZOS

        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def detect_objects(self):
        # Rozpoznání objektů v obrázku
        if hasattr(self, "image"):
            results = self.model(self.image)
            # Získání prvního obrázku z výsledků
            if results and len(results) > 0:  # Kontrola, zda jsou detekce
                detected_image = results.render()[0]  # Získání prvního rámce s detekovanými objekty
                self.display_image(detected_image)  # Zobrazení detekovaného obrázku
            else:
                messagebox.showerror("Chyba", "Žádné objekty nebyly detekovány.")
        else:
            messagebox.showerror("Chyba", "Nejdříve načtěte obrázek nebo video!")

    def edit_image(self):
        # Funkce pro úpravy obrázku
        if hasattr(self, "image"):
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.image)
            self.image.show()  # Zobrazení upraveného obrázku
        else:
            messagebox.showerror("Chyba", "Nejdříve načtěte obrázek!")
    
    def create_chart(self):
        # Vytvoření grafu pro ovládací panel
        self.fig, self.ax = plt.subplots(figsize=(3, 2), dpi=80)
        self.ax.set_facecolor("#f0f0f0")
        self.ax.set_title("Ukázkový graf", fontsize=8)

    def update_chart(self, data):
        # Aktualizace grafu
        self.ax.clear()
        self.ax.plot(data, color='blue', linewidth=2)
        self.fig.canvas.draw()

    def on_button_press(self, event):
        # Zachycení stisknutí tlačítka pro výběr oblasti
        self.x1, self.y1 = event.x, event.y

    def on_mouse_drag(self, event):
        # Zachycení táhnutí myši pro výběr oblasti
        self.x2, self.y2 = event.x, event.y
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red", tags="rect")

    def on_button_release(self, event):
        # Zachycení uvolnění tlačítka pro dokončení výběru oblasti
        self.x2, self.y2 = event.x, event.y
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red")

# Vytvoření hlavního okna aplikace
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
