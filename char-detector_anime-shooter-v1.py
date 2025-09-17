import tkinter as tk
import mss
import numpy as np
import cv2
from ultralytics import YOLO
import time
import ctypes
import win32gui, win32con, win32api
import math
import keyboard
import threading # <-- Nova biblioteca
import queue     # <-- Nova biblioteca

# --- Configurações (sem mudanças) ---
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.4
PRIORIDADES_DE_ALVO = ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 'enemy_paper_scan', 'legs', 'legs_paper']
MAX_AIM_DISTANCE = 150
OVERLAY_TOGGLE_KEY = 'f1'
AIM_TOGGLE_KEY = 'f2'
ACTION_COOLDOWN_SECONDS = 0.006 # <-- Reduzido para 8ms
AIM_SMOOTHING = 6.0

try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

# --- Filas para comunicação entre Threads ---
frame_queue = queue.Queue(maxsize=1)
results_queue = queue.Queue(maxsize=1)

# --- Funções das Threads ---
def capture_thread(monitor, stop_event):
    """Thread que apenas captura a tela e coloca na fila."""
    with mss.mss() as sct:
        while not stop_event.is_set():
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            
            # Coloca o frame na fila se ela estiver vazia, descartando frames antigos
            if frame_queue.empty():
                frame_queue.put(img)
            time.sleep(0.001) # Pequena pausa para não usar 100% da CPU

def detection_thread(model, stop_event):
    """Thread que pega frames, roda a detecção e coloca os resultados na fila."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1) # Espera até 1s por um frame
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            results = model(img_bgr, verbose=False)

            if results_queue.empty():
                results_queue.put(results)
        except queue.Empty:
            continue

# --- Funções Auxiliares (sem mudanças) ---
def is_box_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

def move_mouse_relative(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

# A classe OverlayWindow continua a mesma
class OverlayWindow:
    # ... (cole a classe OverlayWindow inteira aqui, sem nenhuma mudança) ...
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.transparent_color = 'black'
        self.root.attributes('-transparentcolor', self.transparent_color)
        self.root.config(bg=self.transparent_color)
        self.canvas = tk.Canvas(self.root, bg=self.transparent_color, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.after(100, self.make_window_non_interactive)
    def make_window_non_interactive(self):
        try:
            hwnd = self.root.winfo_id()
            styles = win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOOLWINDOW
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        except Exception as e:
            print(f"[ERRO] Falha ao configurar estilos da janela: {e}")
    def update_boxes(self, all_detections, best_target=None):
        self.canvas.delete("all")
        for detection in all_detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)
            is_best = best_target is not None and detection['box'] is best_target['box']
            color = 'yellow' if is_best else 'lime'
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1, y1 - 10, text=detection['class_name'], fill=color, font=("Arial", 10))
    def set_geometry(self, rect):
        x, y, w, h = rect
        self.root.geometry(f"{w}x{h}+{x}+{y}")


def main():
    print("Iniciando bot com mira hierárquica (v. Final)...")
    print(f"  - Pressione '{OVERLAY_TOGGLE_KEY}' para ligar/desligar o overlay (visualização).")
    print(f"  - Pressione '{AIM_TOGGLE_KEY}' para ligar/desligar a mira automática.")
    print("Pressione Ctrl+C no terminal para sair.")
    
    model = YOLO(MODEL_PATH)
    root = tk.Tk()
    overlay = OverlayWindow(root)
    
    last_action_time = 0
    overlay_enabled = True
    aim_enabled = True
    overlay_key_pressed = False
    aim_key_pressed = False

    with mss.mss() as sct:
        # Configuração da área de captura otimizada
        full_screen_monitor = sct.monitors[1]
        screen_width, screen_height = full_screen_monitor["width"], full_screen_monitor["height"]
        screen_center_x, screen_center_y = screen_width / 2, screen_height / 2
        capture_width = int(MAX_AIM_DISTANCE * 2.5)
        capture_height = int(MAX_AIM_DISTANCE * 2.5)
        capture_x = int(screen_center_x - capture_width / 2)
        capture_y = int(screen_center_y - capture_height / 2)
        monitor = {"top": capture_y, "left": capture_x, "width": capture_width, "height": capture_height}
        overlay.set_geometry((capture_x, capture_y, capture_width, capture_height))

        # Inicialização das Threads
        stop_event = threading.Event()
        cap_thread = threading.Thread(target=capture_thread, args=(monitor, stop_event), daemon=True)
        det_thread = threading.Thread(target=detection_thread, args=(model, stop_event), daemon=True)
        cap_thread.start()
        det_thread.start()
        
        all_detections = []
        final_target = None

        while True:
            try:
                # Lógica das teclas de atalho (sem mudanças)
                if keyboard.is_pressed(OVERLAY_TOGGLE_KEY):
                    if not overlay_key_pressed:
                        overlay_enabled = not overlay_enabled
                        print(f"Overlay Visual: {'LIGADO' if overlay_enabled else 'DESLIGADO'}")
                        if overlay_enabled: overlay.root.deiconify()
                        else: overlay.root.withdraw()
                        overlay_key_pressed = True
                else: overlay_key_pressed = False
                if keyboard.is_pressed(AIM_TOGGLE_KEY):
                    if not aim_key_pressed:
                        aim_enabled = not aim_enabled
                        print(f"Mira Automática: {'LIGADA' if aim_enabled else 'DESLIGADA'}")
                        aim_key_pressed = True
                else: aim_key_pressed = False
                
                try:
                    results = results_queue.get_nowait()
                    # (Lógica de processamento de resultados para encontrar final_target - sem mudanças)
                    all_detections = []
                    for result in results:
                        for box in result.boxes:
                            if box.conf[0] > CONFIDENCE_THRESHOLD:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                box_coords = box.xyxy[0]
                                all_detections.append({
                                    'class_name': class_name,
                                    'box': box_coords,
                                    'center': ((box_coords[0] + box_coords[2]) / 2, (box_coords[1] + box_coords[3]) / 2)
                                })
                    targets_in_fov = []
                    capture_center_x, capture_center_y = capture_width / 2, capture_height / 2
                    for det in all_detections:
                        dist = math.sqrt((det['center'][0] - capture_center_x)**2 + (det['center'][1] - capture_center_y)**2)
                        if dist <= MAX_AIM_DISTANCE:
                            det['distance'] = dist
                            targets_in_fov.append(det)
                    base_containers = [tgt for tgt in targets_in_fov if 'enemy' in tgt['class_name']]
                    best_targets_per_container = []
                    for container in base_containers:
                        parts_inside = [part for part in targets_in_fov if 'enemy' not in part['class_name'] and is_box_inside(part['box'], container['box'])]
                        if parts_inside:
                            best_targets_per_container.append(min(parts_inside, key=lambda p: PRIORIDADES_DE_ALVO.index(p['class_name'])))
                        else:
                            best_targets_per_container.append(container)
                    if best_targets_per_container:
                        final_target = min(best_targets_per_container, key=lambda tgt: tgt['distance'])
                    else:
                        final_target = None
                except queue.Empty:
                    pass
                
                # Lógica de mira (sem mudanças)
                if aim_enabled and final_target and (time.time() - last_action_time) > ACTION_COOLDOWN_SECONDS:
                    target_x, target_y = final_target['center']
                    absolute_target_x = target_x + capture_x
                    absolute_target_y = target_y + capture_y
                    current_x, current_y = win32api.GetCursorPos()
                    move_vector_x = absolute_target_x - current_x
                    move_vector_y = absolute_target_y - current_y
                    move_x, move_y = int(move_vector_x / AIM_SMOOTHING), int(move_vector_y / AIM_SMOOTHING)
                    if abs(move_x) > 0 or abs(move_y) > 0:
                        move_mouse_relative(move_x, move_y)
                    last_action_time = time.time()
                
                # --- CORREÇÃO PRINCIPAL AQUI ---
                # Apenas o DESENHO depende do overlay estar ligado
                if overlay_enabled:
                    overlay.update_boxes(all_detections, best_target=final_target)
                
                # Mas o "coração" da GUI (update) roda SEMPRE, para manter o programa responsivo
                overlay.root.update_idletasks()
                overlay.root.update()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ocorreu um erro: {e}")
                time.sleep(1)
                
    stop_event.set()
    root.destroy()

if __name__ == "__main__":
    main()