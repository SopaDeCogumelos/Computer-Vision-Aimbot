import cv2
import os

# --- CONFIGURAÇÕES ---

# Caminho para o seu arquivo de vídeo
VIDEO_PATH = 'Datasets\Dataset-3\Dataset-3.mp4'  # <-- ALTERE AQUI para o nome do seu vídeo

# Nome da pasta onde as imagens serão salvas
OUTPUT_FOLDER = 'Datasets\Dataset-3\dataset_frames' # <-- ALTERE AQUI se desejar

# A cada quantos frames você quer salvar uma imagem.
# Ex: 10 significa que a cada 10 frames do vídeo, 1 será salvo.
# Valores bons para começar: 15, 20, ou 30. Ajuste conforme necessário.
FRAME_SKIP = 30 # <-- AJUSTE ESTE VALOR

# --- CÓDIGO DE EXTRAÇÃO ---

def extrair_frames(video_path, output_folder, frame_skip):
    # Cria a pasta de saída se ela não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta '{output_folder}' criada.")

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o arquivo de vídeo: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print("Iniciando extração de frames...")

    while True:
        # Lê o próximo frame do vídeo
        ret, frame = cap.read()

        # Se 'ret' for False, significa que o vídeo acabou
        if not ret:
            break

        # Verifica se este é um frame que devemos salvar
        if frame_count % frame_skip == 0:
            # Gera um nome de arquivo único para o frame
            # O formato :05d garante que os números tenham 5 dígitos (ex: 00001, 00002)
            # para que os arquivos fiquem ordenados corretamente na pasta.
            file_name = f"frame_{saved_count:05d}.jpg"
            file_path = os.path.join(output_folder, file_name)

            # Salva o frame como uma imagem JPG
            cv2.imwrite(file_path, frame)
            print(f"Salvando: {file_path}")
            saved_count += 1
        
        frame_count += 1

    # Libera o objeto de captura de vídeo
    cap.release()
    print("\nExtração concluída!")
    print(f"Total de {saved_count} frames salvos na pasta '{output_folder}'.")

# Executa a função
if __name__ == "__main__":
    extrair_frames(VIDEO_PATH, OUTPUT_FOLDER, FRAME_SKIP)