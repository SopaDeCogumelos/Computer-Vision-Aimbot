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
MAX_AIM_DISTANCE_SEARCHING = 150
# 2. Raio de foco (em pixels), menor e mais preciso, ativado quando a mira já está em cima de um alvo.
#    Isso evita que a mira "pule" para outros inimigos que apareçam por perto.
MAX_AIM_DISTANCE_FOCUSED = 80
# 3. Define quais tipos de alvo ativam o "Modo Foco". Apenas alvos importantes devem ativá-lo.
FOCUS_TRIGGER_CLASSES = ['head', 'head_paper', 'body', 'body_paper', 'enemy', 'enemy_paper', 'enemy_scan', 
                         'enemy_paper_scan']

# --- Gatilho de Foco Dinâmico ---
FOCUS_TRIGGER_RADIUS = 20  # Raio em pixels da área central que ativa o modo foco (ajuste conforme necessário).

# --- Timer de Histerese para Modo Foco ---
HYSTERESIS_TIME = 0.1  # Tempo em segundos para manter o modo foco após perder o lock no alvo.

# --- Teclas de Atalho e Suavização ---
OVERLAY_TOGGLE_KEY = 'f1'
AIM_TOGGLE_KEY = 'f2'
DEBUG_TOGGLE_KEY = 'f3'  # Tecla para toggle do modo debug (desenho dos raios).

ACTION_COOLDOWN_SECONDS = 0.015 # Tempo de espera entre os movimentos do mouse.

# --- Suavização Dinâmica da Mira ---
AIM_SMOOTHING_SEARCHING = 4.0  # Menor valor para movimentos mais rápidos no modo de busca.
AIM_SMOOTHING_FOCUSED = 8.0    # Maior valor para movimentos mais suaves e precisos no modo foco.

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

# A classe OverlayWindow para desenhar as caixas permanece a mesma.
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

    def update_boxes(self, all_detections, best_target=None, debug_enabled=False, current_max_distance=None, focus_trigger_radius=None):
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
                                        outline='blue', width=2, dash=(4, 4))  # Azul tracejado para raio de mira.

            # Desenha o raio de gatilho do foco (fixo).
            if focus_trigger_radius is not None:
                self.canvas.create_oval(center_x - focus_trigger_radius, center_y - focus_trigger_radius,
                                        center_x + focus_trigger_radius, center_y + focus_trigger_radius,
                                        outline='magenta', width=2, dash=(4, 4))  # Magenta tracejado para gatilho.

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
        self.window.geometry(f"150x90{position}") # Aumentado o tamanho para caber o novo label.

        # Cria os textos (labels) que mostrarão os status.
        self.program_label = tk.Label(self.window, text="BOT: ON", fg="cyan", bg="black", font=("Arial", 10, "bold"))
        self.overlay_label = tk.Label(self.window, text="Overlay: ON", fg="green", bg="black", font=("Arial", 10))
        self.aim_label = tk.Label(self.window, text="Aim Assist: ON", fg="green", bg="black", font=("Arial", 10))
        self.debug_label = tk.Label(self.window, text="Debug: OFF", fg="red", bg="black", font=("Arial", 10))  # Label para debug.

        # Organiza os labels na janela.
        self.program_label.pack(pady=2)
        self.overlay_label.pack()
        self.aim_label.pack()
        self.debug_label.pack()

    def update_status(self, overlay_status, aim_status, debug_status):  # Adicionado debug_status.
        """Atualiza o texto e a cor dos labels com base no estado atual do bot."""
        # Atualiza o status do Overlay (ON/OFF e cor verde/vermelho).
        overlay_text = "Overlay: ON" if overlay_status else "Overlay: OFF"
        overlay_color = "green" if overlay_status else "red"
        self.overlay_label.config(text=overlay_text, fg=overlay_color)

        # Atualiza o status da Mira (ON/OFF e cor verde/vermelho).
        aim_text = "Aim Assist: ON" if aim_status else "Aim Assist: OFF"
        aim_color = "green" if aim_status else "red"
        self.aim_label.config(text=aim_text, fg=aim_color)

        # Atualiza o status do Debug.
        debug_text = "Debug: ON" if debug_status else "Debug: OFF"
        debug_color = "green" if debug_status else "red"
        self.debug_label.config(text=debug_text, fg=debug_color)

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
    debug_enabled = False  # Inicializa debug desativado.
    overlay_key_pressed = False
    aim_key_pressed = False
    debug_key_pressed = False
    last_focus_time = 0  # Tempo da última ativação do modo foco.
    
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
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)  # Inclui debug.
                        if overlay_enabled: overlay.root.deiconify()
                        else: overlay.root.withdraw()
                        overlay_key_pressed = True
                else: overlay_key_pressed = False

                if keyboard.is_pressed(AIM_TOGGLE_KEY):
                    if not aim_key_pressed:
                        aim_enabled = not aim_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)  # Inclui debug.
                        aim_key_pressed = True
                else: aim_key_pressed = False

                # Toggle para debug.
                if keyboard.is_pressed(DEBUG_TOGGLE_KEY):
                    if not debug_key_pressed:
                        debug_enabled = not debug_enabled
                        status_indicator.update_status(overlay_enabled, aim_enabled, debug_enabled)
                        debug_key_pressed = True
                else: debug_key_pressed = False
                
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
                        current_max_distance = MAX_AIM_DISTANCE_SEARCHING  # Começa com o raio de busca.
                        capture_center_point = (capture_width / 2, capture_height / 2)

                        # Verifica se há um alvo importante dentro da área de gatilho central (ativa foco imediatamente).
                        is_currently_focused = False
                        for det in all_detections:
                            if det['class_name'] in FOCUS_TRIGGER_CLASSES:
                                # Calcula a distância do centro do alvo ao centro da tela.
                                dist_to_center = math.sqrt((det['center'][0] - capture_center_point[0])**2 + (det['center'][1] - capture_center_point[1])**2)
                                if dist_to_center <= FOCUS_TRIGGER_RADIUS:
                                    is_currently_focused = True
                                    last_focus_time = time.time()  # Atualiza o timer sempre que o foco é confirmado.
                                    break  # Para a verificação assim que encontra o primeiro alvo qualificado.

                        # Aplica a lógica de histerese se não estiver focado agora, mas o timer ainda não expirou.
                        if is_currently_focused or (time.time() - last_focus_time < HYSTERESIS_TIME):
                            current_max_distance = MAX_AIM_DISTANCE_FOCUSED

                        # Define a suavização com base no modo (busca ou foco).
                        current_smoothing = AIM_SMOOTHING_SEARCHING if current_max_distance == MAX_AIM_DISTANCE_SEARCHING else AIM_SMOOTHING_FOCUSED

                        # 2. Filtra os alvos usando o raio de mira definido (busca ou foco).
                        targets_in_fov = []
                        for det in all_detections:
                            dist = math.sqrt((det['center'][0] - capture_center_point[0])**2 + (det['center'][1] - capture_center_point[1])**2)
                            if dist <= current_max_distance:
                                det['distance'] = dist
                                targets_in_fov.append(det)

                        # 3. Lógica de prioridade hierárquica.
                        base_containers = [tgt for tgt in targets_in_fov if 'enemy' in tgt['class_name']]
                        best_targets_per_container = []
                        for container in base_containers:
                            parts_inside = [part for part in targets_in_fov if 'enemy' not in part['class_name'] and is_box_inside(part['box'], container['box'])]
                            if parts_inside:
                                best_targets_per_container.append(min(parts_inside, key=lambda p: PRIORIDADES_DE_ALVO.index(p['class_name'])))
                            else:
                                best_targets_per_container.append(container)
                        
                        # NOVA: No modo foco, adiciona head/body como alvos independentes se intersectarem a área de gatilho (ignora necessidade de container 'enemy').
                        if current_max_distance == MAX_AIM_DISTANCE_FOCUSED:
                            additional_targets = [det for det in all_detections 
                                                  if det['class_name'] in ['head', 'head_paper', 'body', 'body_paper'] 
                                                  and does_box_intersect_circle(det['box'], capture_center_point, FOCUS_TRIGGER_RADIUS)]
                            # Adiciona à lista, calculando distância se necessário (já que eles não estão em targets_in_fov necessariamente).
                            for tgt in additional_targets:
                                if 'distance' not in tgt:
                                    tgt['distance'] = math.sqrt((tgt['center'][0] - capture_center_point[0])**2 + (tgt['center'][1] - capture_center_point[1])**2)
                            best_targets_per_container.extend(additional_targets)
                        
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
                    move_x = int(move_vector_x / current_smoothing)
                    move_y = int(move_vector_y / current_smoothing)
                    if abs(move_x) > 0 or abs(move_y) > 0:
                        move_mouse_relative(move_x, move_y)
                    last_action_time = time.time()
                
                # --- Atualização da GUI ---
                if overlay_enabled:
                    # Passa parâmetros de debug para update_boxes.
                    overlay.update_boxes(all_detections, best_target=final_target, debug_enabled=debug_enabled,
                                         current_max_distance=current_max_distance if 'current_max_distance' in locals() else None,
                                         focus_trigger_radius=FOCUS_TRIGGER_RADIUS)
                
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