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

# --- Configurações ---
MODEL_PATH = 'best.pt'                 # Caminho para o arquivo do modelo de IA.
CONFIDENCE_THRESHOLD = 0.5             # Confiança mínima (50%) para a IA considerar uma detecção.
PRIORIDADES_DE_ALVO = ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 
                       'enemy_paper_scan', 'legs', 'legs_paper'] # 🎯 Ordem de preferência dos alvos.

# --- NOVA LÓGICA DE RAIO DE MIRA DINÂMICO ---
# 1. Raio de busca (em pixels) usado quando nenhum alvo está na mira.
MAX_AIM_DISTANCE_SEARCHING = 120
# 2. Raio de foco (em pixels), menor e mais preciso, ativado quando a mira já está em cima de um alvo.
#    Isso evita que a mira "pule" para outros inimigos que apareçam por perto.
MAX_AIM_DISTANCE_FOCUSED = 30
# 3. Define quais tipos de alvo ativam o "Modo Foco". Apenas alvos importantes devem ativá-lo.
FOCUS_TRIGGER_CLASSES = ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 
                         'enemy_paper_scan']

# --- Teclas de Atalho e Suavização ---
OVERLAY_TOGGLE_KEY = 'f1'
AIM_TOGGLE_KEY = 'f2'
ACTION_COOLDOWN_SECONDS = 0.006 # Tempo de espera entre os movimentos do mouse.
AIM_SMOOTHING = 9.0             # Suavização da mira (quanto maior, mais suave/lento).

# Garante que o programa funcione corretamente em telas com diferentes escalas de DPI no Windows.
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

# Filas para comunicação segura entre as threads (sem mudanças).
frame_queue = queue.Queue(maxsize=1)
results_queue = queue.Queue(maxsize=1)

# --- Funções das Threads (sem mudanças) ---
# Esta thread apenas captura a tela o mais rápido possível.
def capture_thread(monitor, stop_event):
    with mss.mss() as sct:
        while not stop_event.is_set():
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            if frame_queue.empty():
                frame_queue.put(img)
            time.sleep(0.001)

# Esta thread apenas executa o modelo de IA nas imagens capturadas.
def detection_thread(model, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            results = model(img_bgr, verbose=False)
            if results_queue.empty():
                results_queue.put(results)
        except queue.Empty:
            continue

# --- Funções Auxiliares ---
# Verifica se uma caixa (ex: 'cabeça') está dentro de outra (ex: 'inimigo').
def is_box_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

# --- NOVA FUNÇÃO AUXILIAR ---
def is_point_inside(point, box):
    """Verifica se um ponto (px, py), como o centro da tela, está dentro de uma caixa de detecção."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

# Move o mouse de forma relativa à sua posição atual.
def move_mouse_relative(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

# A classe OverlayWindow para desenhar as caixas permanece a mesma.
class OverlayWindow:
    # (código da classe sem mudanças)
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


# --- INDICADOR DE STATUS ---
class StatusIndicator:
    """Cria e gerencia uma pequena janela para mostrar o status do bot."""
    def __init__(self, root, position="+10+10"):
        # Cria uma janela Toplevel, que é uma janela secundária, independente.
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)      # Remove bordas e barra de título.
        self.window.attributes('-topmost', True) # Sempre no topo.
        self.window.config(bg='black')
        self.window.attributes('-alpha', 0.7)   # Adiciona 60% de transparência.
        self.window.geometry(f"150x70{position}") # Define o tamanho e a posição na tela.

        # Cria os textos (labels) que mostrarão os status.
        self.program_label = tk.Label(self.window, text="BOT: ON", fg="cyan", bg="black", font=("Arial", 10, "bold"))
        self.overlay_label = tk.Label(self.window, text="Overlay: ON", fg="green", bg="black", font=("Arial", 10))
        self.aim_label = tk.Label(self.window, text="Aim Assist: ON", fg="green", bg="black", font=("Arial", 10))

        # Organiza os labels na janela.
        self.program_label.pack(pady=2)
        self.overlay_label.pack()
        self.aim_label.pack()

    def update_status(self, overlay_status, aim_status):
        """Atualiza o texto e a cor dos labels com base no estado atual do bot."""
        # Atualiza o status do Overlay (ON/OFF e cor verde/vermelho).
        overlay_text = "Overlay: ON" if overlay_status else "Overlay: OFF"
        overlay_color = "green" if overlay_status else "red"
        self.overlay_label.config(text=overlay_text, fg=overlay_color)

        # Atualiza o status da Mira (ON/OFF e cor verde/vermelho).
        aim_text = "Aim Assist: ON" if aim_status else "Aim Assist: OFF"
        aim_color = "green" if aim_status else "red"
        self.aim_label.config(text=aim_text, fg=aim_color)

# --- Função Principal ---
def main():
    print("Iniciando bot com mira dinâmica...")
    model = YOLO(MODEL_PATH)
    
    # --- Nova Inicialização da GUI ---
    root = tk.Tk()
    root.withdraw() # Esconde a janela principal e inútil do Tkinter.
    # Cria as janelas secundárias (Toplevel) para o overlay e o indicador.
    overlay = OverlayWindow(tk.Toplevel(root))
    status_indicator = StatusIndicator(tk.Toplevel(root), position="-170+10") # Posição no canto superior direito.

    # --- Inicialização de Variáveis ---
    last_action_time = 0
    overlay_enabled = True
    aim_enabled = True
    overlay_key_pressed = False
    aim_key_pressed = False
    
    # --- Configuração da Captura de Tela ---
    with mss.mss() as sct:
        full_screen_monitor = sct.monitors[1]
        screen_width, screen_height = full_screen_monitor["width"], full_screen_monitor["height"]
        screen_center_x, screen_center_y = screen_width / 2, screen_height / 2
        # A área de captura agora é baseada no raio de BUSCA, para ter um campo de visão maior.
        capture_width = int(MAX_AIM_DISTANCE_SEARCHING * 2.5)
        capture_height = int(MAX_AIM_DISTANCE_SEARCHING * 2.5)
        capture_x = int(screen_center_x - capture_width / 2)
        capture_y = int(screen_center_y - capture_height / 2)
        monitor = {"top": capture_y, "left": capture_x, "width": capture_width, "height": capture_height}
        overlay.set_geometry((capture_x, capture_y, capture_width, capture_height))
        
        # Inicia as threads de captura e detecção (sem mudanças).
        stop_event = threading.Event()
        cap_thread = threading.Thread(target=capture_thread, args=(monitor, stop_event), daemon=True)
        det_thread = threading.Thread(target=detection_thread, args=(model, stop_event), daemon=True)
        cap_thread.start()
        det_thread.start()

        all_detections = []
        final_target = None
        
        # --- Loop Principal ---
        while True:
            try:
                # --- Lógica de Teclas de Atalho (Atualizada) ---
                # Agora, ao pressionar as teclas, a função `update_status` é chamada.
                if keyboard.is_pressed(OVERLAY_TOGGLE_KEY):
                    if not overlay_key_pressed:
                        overlay_enabled = not overlay_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled) # ATUALIZA A JANELA DE STATUS
                        if overlay_enabled: overlay.root.deiconify()
                        else: overlay.root.withdraw()
                        overlay_key_pressed = True
                else: overlay_key_pressed = False

                if keyboard.is_pressed(AIM_TOGGLE_KEY):
                    if not aim_key_pressed:
                        aim_enabled = not aim_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled) # ATUALIZA A JANELA DE STATUS
                        aim_key_pressed = True
                else: aim_key_pressed = False
                
                # --- Processamento dos Resultados ---
                try:
                    results = results_queue.get_nowait()
                    all_detections = [] # Limpa detecções antigas.
                    for result in results:
                        for box in result.boxes:
                            if box.conf[0] > CONFIDENCE_THRESHOLD:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                box_coords = box.xyxy[0]
                                all_detections.append({'class_name': class_name, 'box': box_coords, 'center': ((box_coords[0] + box_coords[2]) / 2, (box_coords[1] + box_coords[3]) / 2)})
                    
                    # --- LÓGICA DE MIRA HIERÁRQUICA E DINÂMICA ---
                    final_target = None
                    if aim_enabled and all_detections:
                        # 1. Decide qual raio de mira usar neste frame.
                        current_max_distance = MAX_AIM_DISTANCE_SEARCHING # Começa com o raio de busca.
                        capture_center_point = (capture_width / 2, capture_height / 2)
                        
                        # Verifica se o centro da tela já está dentro de um alvo importante.
                        for det in all_detections:
                            if det['class_name'] in FOCUS_TRIGGER_CLASSES and is_point_inside(capture_center_point, det['box']):
                                # Se estiver, ativa o "Modo Foco" com o raio menor.
                                current_max_distance = MAX_AIM_DISTANCE_FOCUSED
                                break # Para a verificação assim que encontra o primeiro alvo.

                        # 2. Filtra os alvos usando o raio de mira definido (busca ou foco).
                        targets_in_fov = []
                        for det in all_detections:
                            dist = math.sqrt((det['center'][0] - capture_center_point[0])**2 + (det['center'][1] - capture_center_point[1])**2)
                            if dist <= current_max_distance:
                                det['distance'] = dist
                                targets_in_fov.append(det)

                        # 3. Lógica de prioridade hierárquica (igual à anterior, mas com a lista já filtrada).
                        base_containers = [tgt for tgt in targets_in_fov if 'enemy' in tgt['class_name']]
                        best_targets_per_container = []
                        for container in base_containers:
                            parts_inside = [part for part in targets_in_fov if 'enemy' not in part['class_name'] and is_box_inside(part['box'], container['box'])]
                            if parts_inside:
                                best_targets_per_container.append(min(parts_inside, key=lambda p: PRIORIDADES_DE_ALVO.index(p['class_name'])))
                            else:
                                best_targets_per_container.append(container)
                        
                        # Escolhe o melhor alvo final com base na distância.
                        if best_targets_per_container:
                            final_target = min(best_targets_per_container, key=lambda tgt: tgt['distance'])

                except queue.Empty:
                    pass
                
                # --- Lógica de Ação (Mira) ---
                if aim_enabled and final_target and (time.time() - last_action_time) > ACTION_COOLDOWN_SECONDS:
                    target_x, target_y = final_target['center']
                    move_vector_x = target_x - (capture_width / 2)
                    move_vector_y = target_y - (capture_height / 2)
                    move_x = int(move_vector_x / AIM_SMOOTHING)
                    move_y = int(move_vector_y / AIM_SMOOTHING)
                    if abs(move_x) > 0 or abs(move_y) > 0:
                        move_mouse_relative(move_x, move_y)
                    last_action_time = time.time()
                
                # --- Atualização da GUI ---
                if overlay_enabled:
                    overlay.update_boxes(all_detections, best_target=final_target)
                
                # A chamada `root.update()` agora gerencia TODAS as janelas do Tkinter (principal, overlay, status).
                root.update()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ocorreu um erro: {e}")
                time.sleep(1)
                
    # --- Finalização do Programa ---
    print("Finalizando...")
    stop_event.set() # Sinaliza para as threads pararem.
    root.destroy()   # Fecha todas as janelas do Tkinter.

# Ponto de entrada padrão do Python.
if __name__ == "__main__":
    main()