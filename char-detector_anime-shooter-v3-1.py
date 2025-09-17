# Importa bibliotecas necessárias para o projeto
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
import threading
import queue
import torch  # Adicionado para verificação de GPU
import json  # Adicionado para configurações dinâmicas
import logging  # Adicionado para debug avançado
import unittest  # Adicionado para testes unitários

# Configura logging para debug avançado
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurações Defaults (serão sobrescritas por config.json se existir) ---
DEFAULT_CONFIG = {
    "MODEL_PATH": 'best.pt',
    "CONFIDENCE_THRESHOLD": 0.5,
    "PRIORIDADES_DE_ALVO": ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 
                            'enemy_paper_scan', 'legs', 'legs_paper'],
    "MAX_AIM_DISTANCE_SEARCHING": 160,
    "MAX_AIM_DISTANCE_FOCUSED": 70,
    "FOCUS_TRIGGER_CLASSES": ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 
                              'enemy_paper_scan'],
    "FOCUS_TRIGGER_RADIUS": 20,
    "HYSTERESIS_TIME": 0.15,
    "OVERLAY_TOGGLE_KEY": 'f1',
    "AIM_TOGGLE_KEY": 'f2',
    "DEBUG_TOGGLE_KEY": 'f3',
    "ACTION_COOLDOWN_SECONDS": 0.015,
    "AIM_SMOOTHING_SEARCHING": 4.5,
    "AIM_SMOOTHING_FOCUSED": 6.0,
    "QUEUE_MAXSIZE": 5  # Aumentado para otimização
}

# Função para carregar configurações de JSON
def load_config(file_path='config.json'):
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        logging.info("Configurações carregadas de config.json")
        return config
    except FileNotFoundError:
        logging.warning("config.json não encontrado, usando defaults e criando arquivo.")
        with open(file_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    except json.JSONDecodeError:
        logging.error("Erro ao decodificar config.json, usando defaults.")
        return DEFAULT_CONFIG

# --- Filas para comunicação segura entre as threads ---
def create_queues(maxsize):
    return queue.Queue(maxsize=maxsize), queue.Queue(maxsize=maxsize)

# --- Funções das Threads ---
# Esta thread apenas captura a tela o mais rápido possível.
def capture_thread(monitor, stop_event, frame_queue):
    with mss.mss() as sct:
        while not stop_event.is_set():
            start_time = time.time()
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            if frame_queue.full():
                frame_queue.get()  # Remove o mais antigo se cheio
            frame_queue.put((img, start_time))  # Adiciona timestamp para latência
            time.sleep(0.007)

# Esta thread apenas executa o modelo de IA nas imagens capturadas.
def detection_thread(model, stop_event, frame_queue, results_queue):
    while not stop_event.is_set():
        try:
            frame_data = frame_queue.get(timeout=1)
            img, capture_time = frame_data
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            start_detect = time.time()
            results = model(img_bgr, verbose=False)
            latency = time.time() - capture_time
            logging.debug(f"Latência de detecção: {latency:.4f}s")
            if results_queue.full():
                results_queue.get()
            results_queue.put(results)
        except queue.Empty:
            continue

# --- Funções Auxiliares ---
# Verifica se uma caixa (ex: 'cabeça') está dentro de outra (ex: 'inimigo').
def is_box_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

# Verifica se um ponto (px, py), como o centro da tela, está dentro de uma caixa de detecção.
def is_point_inside(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

# Move o mouse de forma relativa à sua posição atual.
def move_mouse_relative(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

# NOVA: Verifica se uma box intersecta um círculo (para checar interseção com a área de gatilho).
def does_box_intersect_circle(box, center, radius):
    x1, y1, x2, y2 = box
    cx, cy = center
    # Encontra o ponto mais próximo na box ao centro do círculo.
    closest_x = max(x1, min(cx, x2))
    closest_y = max(y1, min(cy, y2))
    # Calcula a distância ao ponto mais próximo.
    dist = math.sqrt((closest_x - cx)**2 + (closest_y - cy)**2)
    return dist <= radius

# A classe OverlayWindow para desenhar as caixas, agora com vetor de movimento.
class OverlayWindow:
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

    def update_boxes(self, all_detections, best_target=None, debug_enabled=False, 
                     current_max_distance=None, focus_trigger_radius=None, 
                     move_vector=None):
        self.canvas.delete("all")
        for detection in all_detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)
            is_best = best_target is not None and detection['box'] is best_target['box']
            color = 'yellow' if is_best else 'lime'
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1, y1 - 10, text=detection['class_name'], fill=color, font=("Arial", 10))

        # Desenho dos raios de debug (círculos) se ativado.
        if debug_enabled:
            center_x = self.canvas.winfo_width() / 2
            center_y = self.canvas.winfo_height() / 2

            # Desenha o raio atual de mira (dinâmico: busca ou foco).
            if current_max_distance is not None:
                self.canvas.create_oval(center_x - current_max_distance, center_y - current_max_distance,
                                        center_x + current_max_distance, center_y + current_max_distance,
                                        outline='blue', width=2, dash=(4, 4))

            # Desenha o raio de gatilho do foco (fixo).
            if focus_trigger_radius is not None:
                self.canvas.create_oval(center_x - focus_trigger_radius, center_y - focus_trigger_radius,
                                        center_x + focus_trigger_radius, center_y + focus_trigger_radius,
                                        outline='magenta', width=2, dash=(4, 4))

            # Desenha vetor de movimento se disponível
            if move_vector:
                vx, vy = move_vector
                self.canvas.create_line(center_x, center_y, center_x + vx, center_y + vy, 
                                        fill='red', width=2, arrow=tk.LAST)

    def set_geometry(self, rect):
        x, y, w, h = rect
        self.root.geometry(f"{w}x{h}+{x}+{y}")

# --- INDICADOR DE STATUS ---
class StatusIndicator:
    """Cria e gerencia uma pequena janela para mostrar o status do bot."""
    def __init__(self, root, position="+10+10"):
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.config(bg='black')
        self.window.attributes('-alpha', 0.7)
        self.window.geometry(f"150x90{position}")

        self.program_label = tk.Label(self.window, text="BOT: ON", fg="cyan", bg="black", font=("Arial", 10, "bold"))
        self.overlay_label = tk.Label(self.window, text="Overlay: ON", fg="green", bg="black", font=("Arial", 10))
        self.aim_label = tk.Label(self.window, text="Aim Assist: ON", fg="green", bg="black", font=("Arial", 10))
        self.debug_label = tk.Label(self.window, text="Debug: OFF", fg="red", bg="black", font=("Arial", 10))

        self.program_label.pack(pady=2)
        self.overlay_label.pack()
        self.aim_label.pack()
        self.debug_label.pack()

    def update_status(self, overlay_status, aim_status, debug_status):
        overlay_text = "Overlay: ON" if overlay_status else "Overlay: OFF"
        overlay_color = "green" if overlay_status else "red"
        self.overlay_label.config(text=overlay_text, fg=overlay_color)

        aim_text = "Aim Assist: ON" if aim_status else "Aim Assist: OFF"
        aim_color = "green" if aim_status else "red"
        self.aim_label.config(text=aim_text, fg=aim_color)

        debug_text = "Debug: ON" if debug_status else "Debug: OFF"
        debug_color = "green" if debug_status else "red"
        self.debug_label.config(text=debug_text, fg=debug_color)

# --- Função Principal ---
def main():
    print("Iniciando bot com mira dinâmica...")
    
    # Carrega configurações dinâmicas
    config = load_config()
    MODEL_PATH = config['MODEL_PATH']
    CONFIDENCE_THRESHOLD = config['CONFIDENCE_THRESHOLD']
    PRIORIDADES_DE_ALVO = config['PRIORIDADES_DE_ALVO']
    MAX_AIM_DISTANCE_SEARCHING = config['MAX_AIM_DISTANCE_SEARCHING']
    MAX_AIM_DISTANCE_FOCUSED = config['MAX_AIM_DISTANCE_FOCUSED']
    FOCUS_TRIGGER_CLASSES = config['FOCUS_TRIGGER_CLASSES']
    FOCUS_TRIGGER_RADIUS = config['FOCUS_TRIGGER_RADIUS']
    HYSTERESIS_TIME = config['HYSTERESIS_TIME']
    OVERLAY_TOGGLE_KEY = config['OVERLAY_TOGGLE_KEY']
    AIM_TOGGLE_KEY = config['AIM_TOGGLE_KEY']
    DEBUG_TOGGLE_KEY = config['DEBUG_TOGGLE_KEY']
    ACTION_COOLDOWN_SECONDS = config['ACTION_COOLDOWN_SECONDS']
    AIM_SMOOTHING_SEARCHING = config['AIM_SMOOTHING_SEARCHING']
    AIM_SMOOTHING_FOCUSED = config['AIM_SMOOTHING_FOCUSED']
    QUEUE_MAXSIZE = config['QUEUE_MAXSIZE']

    # Otimização: Usa GPU se disponível
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    model = YOLO(MODEL_PATH)
    model.to(device)  # Move o modelo para o dispositivo selecionado
    
    # --- Nova Inicialização da GUI ---
    root = tk.Tk()
    root.withdraw()
    overlay = OverlayWindow(tk.Toplevel(root))
    status_indicator = StatusIndicator(tk.Toplevel(root), position="-170+10")

    # --- Inicialização de Variáveis ---
    last_action_time = 0
    overlay_enabled = True
    aim_enabled = True
    debug_enabled = False
    overlay_key_pressed = False
    aim_key_pressed = False
    debug_key_pressed = False
    last_focus_time = 0
    last_frame_time = time.time()  # Para cálculo de FPS
    
    # --- Configuração da Captura de Tela ---
    with mss.mss() as sct:
        full_screen_monitor = sct.monitors[1]
        screen_width, screen_height = full_screen_monitor["width"], full_screen_monitor["height"]
        screen_center_x, screen_center_y = screen_width / 2, screen_height / 2
        capture_width = int(MAX_AIM_DISTANCE_SEARCHING * 2.5)
        capture_height = int(MAX_AIM_DISTANCE_SEARCHING * 2.5)
        capture_x = int(screen_center_x - capture_width / 2)
        capture_y = int(screen_center_y - capture_height / 2)
        monitor = {"top": capture_y, "left": capture_x, "width": capture_width, "height": capture_height}
        overlay.set_geometry((capture_x, capture_y, capture_width, capture_height))
        
        # Cria queues com maxsize otimizado
        frame_queue, results_queue = create_queues(QUEUE_MAXSIZE)
        
        # Inicia as threads de captura e detecção (adaptadas para queues e logging)
        stop_event = threading.Event()
        cap_thread = threading.Thread(target=capture_thread, args=(monitor, stop_event, frame_queue), daemon=True)
        det_thread = threading.Thread(target=detection_thread, args=(model, stop_event, frame_queue, results_queue), daemon=True)
        cap_thread.start()
        det_thread.start()

        all_detections = []
        final_target = None
        move_vector = None  # Para debug de vetor
        
        # --- Loop Principal ---
        while True:
            try:
                current_time = time.time()
                fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                logging.debug(f"FPS: {fps:.2f}")
                last_frame_time = current_time

                # --- Lógica de Teclas de Atalho ---
                if keyboard.is_pressed(OVERLAY_TOGGLE_KEY):
                    if not overlay_key_pressed:
                        overlay_enabled = not overlay_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)
                        if overlay_enabled: overlay.root.deiconify()
                        else: overlay.root.withdraw()
                        overlay_key_pressed = True
                else: overlay_key_pressed = False

                if keyboard.is_pressed(AIM_TOGGLE_KEY):
                    if not aim_key_pressed:
                        aim_enabled = not aim_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)
                        aim_key_pressed = True
                else: aim_key_pressed = False

                if keyboard.is_pressed(DEBUG_TOGGLE_KEY):
                    if not debug_key_pressed:
                        debug_enabled = not debug_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)
                        debug_key_pressed = True
                else: debug_key_pressed = False
                
                # --- Processamento dos Resultados ---
                try:
                    results = results_queue.get_nowait()
                    all_detections = []
                    for result in results:
                        for box in result.boxes:
                            if box.conf[0].item() > CONFIDENCE_THRESHOLD:
                                class_id = int(box.cls[0].item())
                                class_name = model.names[class_id]
                                box_coords = box.xyxy[0].cpu().tolist()
                                center = ((box_coords[0] + box_coords[2]) / 2, (box_coords[1] + box_coords[3]) / 2)
                                all_detections.append({'class_name': class_name, 'box': box_coords, 'center': center})
                    
                    # --- LÓGICA DE MIRA HIERÁRQUICA E DINÂMICA ---
                    final_target = None
                    if aim_enabled and all_detections:
                        current_max_distance = MAX_AIM_DISTANCE_SEARCHING
                        capture_center_point = (capture_width / 2, capture_height / 2)

                        is_currently_focused = False
                        for det in all_detections:
                            if det['class_name'] in FOCUS_TRIGGER_CLASSES:
                                dist_to_center = math.sqrt((det['center'][0] - capture_center_point[0])**2 + (det['center'][1] - capture_center_point[1])**2)
                                if dist_to_center <= FOCUS_TRIGGER_RADIUS:
                                    is_currently_focused = True
                                    last_focus_time = time.time()
                                    break

                        if is_currently_focused or (time.time() - last_focus_time < HYSTERESIS_TIME):
                            current_max_distance = MAX_AIM_DISTANCE_FOCUSED

                        current_smoothing = AIM_SMOOTHING_SEARCHING if current_max_distance == MAX_AIM_DISTANCE_SEARCHING else AIM_SMOOTHING_FOCUSED

                        targets_in_fov = []
                        for det in all_detections:
                            dist = math.sqrt((det['center'][0] - capture_center_point[0])**2 + (det['center'][1] - capture_center_point[1])**2)
                            if dist <= current_max_distance:
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
                        
                        if current_max_distance == MAX_AIM_DISTANCE_FOCUSED:
                            additional_targets = [det for det in all_detections 
                                                  if det['class_name'] in ['head', 'head_paper', 'body', 'body_paper'] 
                                                  and does_box_intersect_circle(det['box'], capture_center_point, (MAX_AIM_DISTANCE_FOCUSED+FOCUS_TRIGGER_RADIUS)/2.0)]
                            for tgt in additional_targets:
                                if 'distance' not in tgt:
                                    tgt['distance'] = math.sqrt((tgt['center'][0] - capture_center_point[0])**2 + (tgt['center'][1] - capture_center_point[1])**2)
                            best_targets_per_container.extend(additional_targets)
                        
                        if best_targets_per_container:
                            final_target = min(best_targets_per_container, key=lambda tgt: tgt['distance'])

                except queue.Empty:
                    pass
                
                # --- Lógica de Ação (Mira) ---
                move_vector = None  # Reset
                if aim_enabled and final_target and (time.time() - last_action_time) > ACTION_COOLDOWN_SECONDS:
                    target_x, target_y = final_target['center']
                    move_vector_x = target_x - (capture_width / 2)
                    move_vector_y = target_y - (capture_height / 2)
                    move_x = int(move_vector_x / current_smoothing)
                    move_y = int(move_vector_y / current_smoothing)
                    move_vector = (move_vector_x, move_vector_y)  # Para debug
                    if abs(move_x) > 0 or abs(move_y) > 0:
                        move_mouse_relative(move_x, move_y)
                    last_action_time = time.time()
                
                # --- Atualização da GUI ---
                if overlay_enabled:
                    overlay.update_boxes(all_detections, best_target=final_target, debug_enabled=debug_enabled,
                                         current_max_distance=current_max_distance if 'current_max_distance' in locals() else None,
                                         focus_trigger_radius=FOCUS_TRIGGER_RADIUS,
                                         move_vector=move_vector)
                
                root.update()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ocorreu um erro: {e}")
                logging.error(f"Erro no loop principal: {e}")
                time.sleep(1)
                
    # --- Finalização do Programa ---
    print("Finalizando...")
    stop_event.set()
    root.destroy()

# --- Testes Unitários ---
class TestAuxFunctions(unittest.TestCase):
    def test_is_box_inside(self):
        inner = (2, 2, 8, 8)
        outer = (0, 0, 10, 10)
        self.assertTrue(is_box_inside(inner, outer))
        self.assertFalse(is_box_inside(inner, (3, 3, 7, 7)))

    def test_is_point_inside(self):
        point = (5, 5)
        box = (0, 0, 10, 10)
        self.assertTrue(is_point_inside(point, box))
        self.assertFalse(is_point_inside(point, (6, 6, 9, 9)))

    def test_does_box_intersect_circle(self):
        box = (0, 0, 10, 10)
        center = (5, 5)
        radius = 1
        self.assertTrue(does_box_intersect_circle(box, center, radius))
        center = (12, 5)
        radius = 1
        self.assertFalse(does_box_intersect_circle(box, center, radius))

# Ponto de entrada padrão do Python.
if __name__ == "__main__":
    # Para rodar testes: python script.py test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=sys.argv[:1] + sys.argv[2:])
    else:
        main()