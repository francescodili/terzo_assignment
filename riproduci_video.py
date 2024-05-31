import cv2
import os

def play_video_from_frames(folder_path, frame_rate=30):
    # Ottieni la lista dei file nella cartella e ordina per nome
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Verifica che la cartella non sia vuota
    if not frame_files:
        print("No frames found in the folder.")
        return

    # Leggi il primo frame per ottenere dimensioni e tipo
    first_frame_path = os.path.join(folder_path, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Crea una finestra per la riproduzione video
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', width, height)

    # Riproduci i frame come video
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Could not read frame: {frame_file}")
            continue

        cv2.imshow('Video', frame)
        key = cv2.waitKey(int(10000 / frame_rate))  # Attendi per il tempo specificato in millisecondi

        # Premere 'q' per uscire dalla riproduzione
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = r"../bbox/train_bbox/MOT17-02-DPM"  # Sostituisci con il percorso della tua cartella di frame
    play_video_from_frames(folder_path)
