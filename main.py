import numpy as np
import math
from ultralytics import YOLO
import cvzone
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from yolo_cfg import model, model_seg, model_pose
from copy import deepcopy
import pygame
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BG = "#1e1e2e"
BG_VIDEO = "#11111b"
FG = "#061956"
BTN_BG = "#313244"
BTN_ACTIVE = "#ff3232"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
MAX_DISPLAY_WIDTH = 640
MAX_DISPLAY_HEIGHT = 500

class AppFilters:
    def __init__(self, gui):
        self.yolo = model
        self.yolo_seg = model_seg
        self.yolo_pose = model_pose
        self.yolo_every = 5
        self.yolo_count = 0
        self.yolo_detections = []

        self.gui = gui
        self.gui.title("Filtros de PDI - T1")
        self.gui.configure(bg=BG)
        self.gui.resizable(False, False)

        self.mode = "camera"
        
        self.capture = cv2.VideoCapture(0)
        
        self.video_width = VIDEO_WIDTH
        self.video_height = VIDEO_HEIGHT

        self.current_image = None
        self.current_filter = None

        self.tracker = None
        self.tracking = False
        self.selecting = False
        self.selection_start = None
        self.selection_end = None
        self.tracking_box = None

        self.buttons = {}

        self.canny_t1 = tk.StringVar(value="50")
        self.canny_t2 = tk.StringVar(value="150")
        self.median_kernel = tk.StringVar(value="5")
        self.mean_kernel = tk.StringVar(value="5")
        self.erode_iterations = tk.StringVar(value="1")
        self.dilate_iterations = tk.StringVar(value="1")
        self.erode_kernel_rows = tk.StringVar(value="3")
        self.erode_kernel_cols = tk.StringVar(value="3")
        self.dilate_kernel_rows = tk.StringVar(value="3")
        self.dilate_kernel_cols = tk.StringVar(value="3")
        self.abertura_kernel_rows = tk.StringVar(value="3")
        self.abertura_kernel_cols = tk.StringVar(value="3")
        self.fechamento_kernel_rows = tk.StringVar(value="3")
        self.fechamento_kernel_cols = tk.StringVar(value="3")

        self.left_frame = tk.Frame(gui, bg=BG)
        self.left_frame.grid(row=0, column=0, sticky="n", padx=20, pady=20)

        self.filter_frame = tk.Frame(self.left_frame, bg=BG)
        self.filter_frame.pack()

        self.right_frame = tk.Frame(gui, bg=BG)
        self.right_frame.grid(row=0, column=1, sticky="n", padx=(0, 20), pady=20)
        
        self.cap_screen = tk.Canvas(self.right_frame, width=self.video_width, height=self.video_height, bg=BG_VIDEO, highlightthickness=0)

        self.cap_screen.bind("<ButtonPress-1>", self.start_selection)
        self.cap_screen.bind("<B1-Motion>", self.update_selection)
        self.cap_screen.bind("<ButtonRelease-1>", self.finish_selection)

        self.cap_screen.grid(row=0, column=0, columnspan=2)
        
        self.cam_mode = tk.Button(self.right_frame, text="Camera", width=18, height=2, bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE, relief=tk.FLAT, command=lambda: self.set_mode("camera"),)
        self.cam_mode.grid(row=1, column=0, pady=6, sticky="w")
        
        self.image_mode = tk.Button(self.right_frame, text="Imagem", width=18, height=2, bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE, relief=tk.FLAT, command=lambda: self.set_mode("image"),)
        self.image_mode.grid(row=1, column=1, pady=6, sticky="w")

        self.image_path = None
        self.original_image = None
        self.loaded_image = None
        self.filtered_image = None
        
        self.create_buttons_camera()

        self.refresh_cam()
        self.gui.protocol("WM_DELETE_WINDOW", self.close)

    def add_row(self, row, text, filter_name):
        btn = tk.Button(self.filter_frame, text=text, width=18, height=2, bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE, relief=tk.FLAT, command=lambda: self.set_filter(filter_name),)
        btn.grid(row=row, column=0, pady=6, sticky="w")
        self.buttons[filter_name] = btn

        params = tk.Frame(self.filter_frame, bg=BG)
        params.grid(row=row, column=1, padx=(12, 0), sticky="w")
        return params

    def add_param(self, parent, label, variable):
        tk.Label(parent, text=label, bg=BG).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(parent, textvariable=variable, width=6, justify="center").pack(side=tk.LEFT, padx=(0, 12))
    
    def create_buttons_camera(self):
        self.buttons = {}
        self.add_row(0, "Sem filtro", None)
        self.add_row(1, "Niveis de cinza", "gray")
        self.add_row(2, "Negativo", "negative")
        self.add_row(3, "Otsu", "otsu")

        params = self.add_row(4, "Canny", "canny")
        self.add_param(params, "threshold 1:", self.canny_t1)
        self.add_param(params, "threshold 2:", self.canny_t2)

        params = self.add_row(5, "Suavização\nmediana", "median")
        self.add_param(params, "tam. kernel:", self.median_kernel)

        params = self.add_row(6, "Suavização\nmedia", "mean")
        self.add_param(params, "tam. kernel:", self.mean_kernel)

        params = self.add_row(7, "Deteccao de\nobjetos", "yolo")

        params = self.add_row(8, "Erosao", "erode")
        self.add_param(params, "iteracoes", self.erode_iterations)
        self.add_param(params, "rows kernel", self.erode_kernel_rows)
        self.add_param(params, "cols kernel", self.erode_kernel_cols)

        params = self.add_row(9, "Dilatacao", "dilate")
        self.add_param(params, "iteracoes", self.dilate_iterations)
        self.add_param(params, "rows kernel", self.dilate_kernel_rows)
        self.add_param(params, "cols kernel", self.dilate_kernel_cols)

        params = self.add_row(10, "Abertura", "abertura")
        self.add_param(params, "rows kernel", self.abertura_kernel_rows)
        self.add_param(params, "cols kernel", self.abertura_kernel_cols)

        params = self.add_row(11, "Fechamento", "fechamento")
        self.add_param(params, "rows kernel", self.fechamento_kernel_rows)
        self.add_param(params, "cols kernel", self.fechamento_kernel_cols)

        self.add_row(12, "Rastreamento\nde objeto", "tracking")

        self.add_row(13, "Segmentacao\nde objetos", "yolo_seg")

        self.add_row(14, "Deteccao\nde poses", "yolo_pose")

    def create_buttons_image(self):
        self.add_row(0, "Sem filtro", None)
        self.add_row(1, "Niveis de cinza", "gray")
        self.add_row(2, "Negativo", "negative")
        self.add_row(3, "Otsu", "otsu")

        params = self.add_row(4, "Suavização\nmediana", "median")
        self.add_param(params, "tam. kernel:", self.median_kernel)

        params = self.add_row(5, "Suavização\nmedia", "mean")
        self.add_param(params, "tam. kernel:", self.mean_kernel)

        params = self.add_row(6, "Canny", "canny")
        self.add_param(params, "threshold 1:", self.canny_t1)
        self.add_param(params, "threshold 2:", self.canny_t2)

        params = self.add_row(7, "Erosao", "erode")
        self.add_param(params, "iteracoes", self.erode_iterations)
        self.add_param(params, "rows kernel", self.erode_kernel_rows)
        self.add_param(params, "cols kernel", self.erode_kernel_cols)

        params = self.add_row(8, "Dilatacao", "dilate")
        self.add_param(params, "iteracoes", self.dilate_iterations)
        self.add_param(params, "rows kernel", self.dilate_kernel_rows)
        self.add_param(params, "cols kernel", self.dilate_kernel_cols)

        params = self.add_row(9, "Abertura", "abertura")
        self.add_param(params, "rows kernel", self.abertura_kernel_rows)
        self.add_param(params, "cols kernel", self.abertura_kernel_cols)

        params = self.add_row(10, "Fechamento", "fechamento")
        self.add_param(params, "rows kernel", self.fechamento_kernel_rows)
        self.add_param(params, "cols kernel", self.fechamento_kernel_cols)

        self.add_row(11, "Histograma", "histogram")

        self.add_row(12, "Componentes\nconexos", "components")

        self.add_row(13, "Medidas\nobjetos", "measurements")

    def set_filter(self, filter_name):
        self.current_filter = filter_name
        self.yolo_detections = []
        self.yolo_count = 0

        self.tracker = None
        self.tracking = False
        self.tracking_box = None
        self.cap_screen.delete("selection")

        for name, btn in self.buttons.items():
            btn.configure(bg=BTN_ACTIVE if name == filter_name else BTN_BG, fg=BG if name == filter_name else FG)

        if self.mode == "image" and self.loaded_image is not None:
            self.filtered_image = self.apply_filter_image(self.loaded_image.copy())

    def set_mode(self, mode):
        self.mode = mode
        self.filter_frame.destroy()
        self.filter_frame = tk.Frame(self.left_frame, bg=BG)
        self.filter_frame.pack()
        self.buttons = {}

        if mode == "image":
            self.create_buttons_image()
            self.select_image()
            self.set_filter(None)
            
        elif mode == "camera":
            self.create_buttons_camera()
            self.video_width = VIDEO_WIDTH
            self.video_height = VIDEO_HEIGHT
            self.cap_screen.config(width=self.video_width, height=self.video_height)
            
        self.current_filter = None

    @staticmethod
    def read_int(var, default, minimum=0):
        try:
            return max(minimum, int(var.get()))
        except ValueError:
            return default

    def odd_kernel(self, var, default=5):
        k = self.read_int(var, default, minimum=3)
        return k if k % 2 == 1 else k + 1

    def refresh_cam(self):
        if self.mode == "camera":
            ret, frame = self.capture.read()

            if not ret:
                self.gui.after(15, self.refresh_cam)
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.apply_filter_camera(frame)
            self.current_image = Image.fromarray(frame)
            image = self.current_image.resize((self.video_width, self.video_height))
            self.current_image = ImageTk.PhotoImage(image)
            self.cap_screen.delete("frame")
            self.cap_screen.create_image(0, 0, anchor=tk.NW, image=self.current_image, tags="frame")
            self.cap_screen.tag_lower("frame")
            self.gui.after(15, self.refresh_cam)

        elif self.mode == "image":
            if self.filtered_image is None:
                self.gui.after(100, self.refresh_cam)
                return

            frame = self.filtered_image
            self.cap_screen.config(width=self.video_width, height=self.video_height)
            self.current_image = Image.fromarray(frame)
            self.current_image = ImageTk.PhotoImage(self.current_image)
            self.cap_screen.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            self.gui.after(100, self.refresh_cam)


    def select_image(self):
        image_path = filedialog.askopenfilename(title="Selecione uma imagem")
        if image_path:
            self.image_path = image_path

            image = cv2.imread(self.image_path)

            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.original_image = rgb_image
                orig_height, orig_width = rgb_image.shape[:2]
                scale = min(MAX_DISPLAY_WIDTH / orig_width, MAX_DISPLAY_HEIGHT / orig_height, 1.0)
                self.video_width = int(orig_width * scale)
                self.video_height = int(orig_height * scale)

                self.loaded_image = cv2.resize(rgb_image, (self.video_width, self.video_height))

    def apply_filter_camera(self, frame):
        f = self.current_filter

        if f == None:
            return frame

        if f == "gray":
            gray_image = frame[:, :, 0]//3 + frame[:, :, 1]//3 + frame[:, :, 2]//3
            return gray_image

        if f == "negative":
            negative_image = 255 - frame
            return negative_image

        if f == "otsu":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rows, cols = frame.shape[:2]
            img_arr = np.array(frame)
            hist, bin_edges = np.histogram(img_arr.flatten(), bins=256, range=(0, 256))
            total_pixels = frame.size

            weights_B = np.cumsum(hist)
            weights_F = total_pixels - weights_B

            t = np.arange(256)
            sum_B = np.cumsum(t * hist)
            total_sum = sum_B[-1]

            valid_mask = (weights_B > 0) & (weights_F > 0)

            mean_B = sum_B[valid_mask] / weights_B[valid_mask]
            mean_F = (total_sum - sum_B[valid_mask]) / weights_F[valid_mask]

            var_between = weights_B[valid_mask].astype(float) * weights_F[valid_mask].astype(float) * (mean_B - mean_F) ** 2

            valid_ts = t[valid_mask]
            threshold = valid_ts[np.argmax(var_between)]

            img_final = np.zeros_like(frame)

            return np.where(frame >= threshold, 255, 0).astype(np.uint8)

        if f == "canny":
            t1 = self.read_int(self.canny_t1, 50)
            t2 = self.read_int(self.canny_t2, 150)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            return cv2.Canny(gray_image, t1, t2)

        if f == "median":
            k = self.odd_kernel(self.median_kernel)
            return cv2.medianBlur(frame, k)

        if f == "mean":
            k = self.odd_kernel(self.mean_kernel)
            return cv2.blur(frame, (k, k))

        if f == "yolo":
            if self.yolo_count % self.yolo_every == 0:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = self.yolo(bgr, conf=0.4, iou=0.5, imgsz=640, verbose=False)

                self.yolo_detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    name = self.yolo.names[int(box.cls[0])]
                    self.yolo_detections.append((x1, y1, x2, y2, name, conf))
            
            self.yolo_count += 1

            phone_detected = any(object[4] == "cell phone" for object in self.yolo_detections)
            clock_detected = any(object[4] == "clock" for object in self.yolo_detections)

            if phone_detected:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            if clock_detected:
                self.close()

            for x1, y1, x2, y2, name, conf in self.yolo_detections:
                cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1))
                cvzone.putTextRect(frame, f"{name} {conf:.2f}", (max(0, x1), max(35, y1)), scale=3, thickness=1)
                
            return frame

        if f == "tracking":
            if self.tracking_box is not None and self.tracker is None:
                frame_h, frame_w = frame.shape[:2]
                sx = frame_w / self.video_width
                sy = frame_h / self.video_height
                x, y, w, h = self.tracking_box
                box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

                self.tracker = self.create_tracker()
                self.tracker.init(frame, box)
                self.tracking = True

            if self.tracking:
                ok, box = self.tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Rastreando", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Objeto perdido", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Desenhe um quadrado no objeto", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            return frame

        if f == "erode":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.erode_kernel_rows, 3), self.read_int(self.erode_kernel_cols, 3)))
            frame = cv2.erode(frame, kernel, iterations=self.read_int(self.erode_iterations, 1))
            return frame

        if f == "dilate":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.dilate_kernel_rows, 3), self.read_int(self.dilate_kernel_cols, 3)))
            frame = cv2.dilate(frame, kernel, iterations=self.read_int(self.dilate_iterations, 1))
            return frame

        if f == "abertura":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.abertura_kernel_rows, 3), self.read_int(self.abertura_kernel_cols, 3)))
            frame = cv2.erode(frame, kernel, iterations=1)
            frame = cv2.dilate(frame, kernel, iterations=1)
            return frame

        if f == "fechamento":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.fechamento_kernel_rows, 3), self.read_int(self.fechamento_kernel_cols, 3)))
            frame = cv2.dilate(frame, kernel, iterations=1)
            frame = cv2.erode(frame, kernel, iterations=1)
            return frame

        if f == "yolo_seg":
            results = model_seg(frame, verbose=False)
            annotaded_frame = results[0].plot()
            return annotaded_frame

        if f == "yolo_pose":
            results = model_pose(frame, verbose=False)
            annotaded_frame = results[0].plot()
            return annotaded_frame

    def apply_filter_image(self, image):
        f = self.current_filter

        if f == None:
            return image

        if f == "gray":
            gray_image = image[:, :, 0]//3 + image[:, :, 1]//3 + image[:, :, 2]//3
            return gray_image

        if f == "negative":
            negative_image = 255 - image
            return negative_image

        if f == "otsu":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_arr = np.array(image)
            hist, bin_edges = np.histogram(img_arr.flatten(), bins=256, range=(0, 256))
            total_pixels = image.size

            weights_B = np.cumsum(hist)
            weights_F = total_pixels - weights_B

            t = np.arange(256)
            sum_B = np.cumsum(t * hist)
            total_sum = sum_B[-1]

            valid_mask = (weights_B > 0) & (weights_F > 0)

            mean_B = sum_B[valid_mask] / weights_B[valid_mask]
            mean_F = (total_sum - sum_B[valid_mask]) / weights_F[valid_mask]

            var_between = weights_B[valid_mask].astype(float) * weights_F[valid_mask].astype(float) * (mean_B - mean_F) ** 2

            valid_ts = t[valid_mask]
            threshold = valid_ts[np.argmax(var_between)]

            return np.where(image >= threshold, 255, 0).astype(np.uint8)

        if f == "canny":
            t1 = self.read_int(self.canny_t1, 50)
            t2 = self.read_int(self.canny_t2, 150)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            return cv2.Canny(gray_image, t1, t2)

        if f == "median":
            tam_janela = self.odd_kernel(self.median_kernel)
            altura, largura = image.shape[:2]
            margem = tam_janela // 2

            image_padded = np.pad(image, ((margem, margem), (margem, margem), (0, 0)), mode="reflect")
            img_transformada = np.zeros_like(image)
        
            for i in range(altura):
                for j in range(largura):
                    vizinhanca = image_padded[i : i + tam_janela, j : j + tam_janela]
        
                    for canal in range(3):
                        img_transformada[i, j, canal] = np.median(vizinhanca[:, :, canal])
        
            img_transformada = img_transformada.astype(np.uint8)
            return img_transformada

        if f == "mean":
            tam_janela = self.odd_kernel(self.mean_kernel)
            margem = tam_janela // 2
            altura, largura = image.shape[:2]

            image_padded = np.pad(image, ((margem, margem), (margem, margem), (0, 0)), mode="reflect")

            img_transformada = np.zeros_like(image)
        
            for i in range(altura):
                for j in range(largura):
                    vizinhanca = image_padded[i : i + tam_janela, j : j + tam_janela]

                    for canal in range(3):
                        img_transformada[i, j, canal] = np.mean(vizinhanca[:, :, canal])
        
            img_transformada = img_transformada.astype(np.uint8)
            return img_transformada

        if f == "erode":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.erode_kernel_rows, 3), self.read_int(self.erode_kernel_cols, 3)))
            image = cv2.erode(image, kernel, iterations=self.read_int(self.erode_iterations, 1))
            return image

        if f == "dilate":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.dilate_kernel_rows, 3), self.read_int(self.dilate_kernel_cols, 3)))
            image = cv2.dilate(image, kernel, iterations=self.read_int(self.dilate_iterations, 1))
            return image

        if f == "abertura":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.abertura_kernel_rows, 3), self.read_int(self.abertura_kernel_cols, 3)))
            image = cv2.erode(image, kernel, iterations=1)
            image = cv2.dilate(image, kernel, iterations=1)
            return image

        if f == "fechamento":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.read_int(self.fechamento_kernel_rows, 3), self.read_int(self.fechamento_kernel_cols, 3)))
            image = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(image, kernel, iterations=1)
            return image

        if f == "histogram":
            qtd_pixels = np.zeros(256)
            height, width = image.shape[:2]

            for i in range(height):
                for j in range(width):
                    qtd_pixels[image[i][j]] += 1

            hist_window = tk.Toplevel(self.gui)
            hist_window.configure(bg=BG)

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            ax.bar(range(256), qtd_pixels, width=1)
            ax.set_title(f'Histograma da imagem carregada')
            ax.set_xlabel("Valores dos pixels")
            ax.set_ylabel("Quantidades dos pixels")

            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            canvas.draw()

            return image

        if f == "components" or f == "measurements":
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            binary_image = np.where(gray_image > 127, 255, 0).astype(np.uint8)
            height, width = binary_image.shape

            labels = np.zeros((height, width), dtype=np.int32)

            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]

            label = 0
            components = []

            for i in range(height):
                for j in range(width):
                    if binary_image[i, j] == 0:
                        continue

                    if labels[i, j] != 0:
                        continue

                    label += 1

                    queue = [(i, j)]
                    labels[i, j] = label

                    pixels = []

                    while queue:
                        y, x = queue.pop(0)

                        pixels.append((y, x))
                        
                        for dy, dx in neighbors:
                            ny = y + dy
                            nx = x + dx

                            if 0 <= ny < height and 0 <= nx < width:
                                if binary_image[ny, nx] > 0 and labels[ny, nx] == 0:
                                    labels[ny, nx] = label
                                    queue.append((ny, nx))
                    components.append(pixels)

            if f == "components":
                result = np.zeros((height, width, 3), dtype=np.uint8)

                rng = np.random.default_rng(42)

                colors = rng.integers(50, 256, size=(len(components), 3))

                for index, component in enumerate(components):
                    color = colors[index]
                    for y, x in component:
                        result[y, x] = color

                return result

            if f == "measurements":
                areas = []
                perimeters = []
                diameters = []
                for component_index, component in enumerate(components):
                    component_label = component_index + 1
                    
                    area = len(component)
                    areas.append(area)

                    perimeter = 0

                    for y, x in component:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny = y + dy
                            nx = x + dx

                            if(ny < 0 or ny >= height or nx < 0 or nx >= width):
                                perimeter += 1
                            elif labels[ny, nx] != component_label:
                                perimeter += 1
                                
                    perimeters.append(perimeter)

                    max_dist = 0
                    for i in range(len(component)):
                        for j in range(i+1, len(component)):
                            y1, x1 = component[i]
                            y2, x2 = component[j]

                            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

                            if distance > max_dist:
                                max_dist = distance

                    diameters.append(max_dist)

                result_window = tk.Toplevel(self.gui)
                result_window.title("Medidas dos objetos")

                main_frame = tk.Frame(result_window)
                main_frame.pack(fill="both", expand=True)

                canvas = tk.Canvas(main_frame)
                canvas.pack(side="left", fill="both", expand=True)

                scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
                scrollbar.pack(side="right", fill="y")

                canvas.configure(yscrollcommand=scrollbar.set)

                results_frame = tk.Frame(canvas)

                canvas_window = canvas.create_window((0, 0), window=results_frame, anchor="nw")
                
                for i in range (len(components)):
                    text = (
                        f"Objeto {i+1}:\n"
                        f"Área: {areas[i]} pixels\n"
                        f"Perímetro: {perimeters[i]} pixels\n"
                        f"Diâmetro: {diameters[i]} pixels"
                    )
                    label = tk.Label(results_frame, text=text, bg=BG, justify=tk.LEFT, anchor="w", font=("Arial", 11))
                    label.pack(fill="x")

                results_frame.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))

                def resize_frame(event):
                    canvas.itemconfig(canvas_window, width=event.width)

                canvas.bind("<Configure>", resize_frame)
                
                return image


    def start_selection(self, event):
        if self.current_filter != "tracking":
            return

        self.selecting = True
        self.selection_start = (event.x, event.y)
        self.selection_end = (event.x, event.y)

    def update_selection(self, event):
        if not self.selecting:
            return

        self.selection_end = (event.x, event.y)
        self.cap_screen.delete("selection")
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end

        self.cap_screen.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="selection")

    def finish_selection(self, event):
        if not self.selecting:
            return

        self.selecting = False
        self.selection_end = (event.x, event.y)

        x1, y1 = self.selection_start
        x2, y2 = self.selection_end

        x = min(x1, x2)
        y = min(y1, y2)

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        self.cap_screen.delete("selection")

        if width < 10 or height < 10:
            return
        
        self.tracking_box = (x, y, width, height)

    def create_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerMIL_create()
        
    def close(self):
        self.capture.release()
        self.gui.destroy()




if __name__ == "__main__":
    pygame.mixer.init()
    pygame.mixer.music.load("payphone.mp3")
    pygame.mixer.music.set_volume(0.7)
    gui = tk.Tk()
    gui.title("Filtros de PDI - T1")

    app = AppFilters(gui)
    gui.mainloop()
