import logging

class Track:
    def __init__(self, track_id, bbox, crop, max_frames_to_skip=5):
        """
        Inizializza una nuova traccia con un ID, una bounding box, un crop e un massimo di frame saltati.
        
        :param track_id: Identificatore unico della traccia.
        :param bbox: Bounding box della traccia.
        :param crop: Crop della traccia.
        :param max_frames_to_skip: Numero massimo di frame che possono essere saltati prima che la traccia venga considerata persa.
        """
        self.track_id = track_id
        self.bbox = bbox
        self.crop = crop
        self.max_frames_to_skip = max_frames_to_skip
        self.skipped_frames = 0

    def update(self, bbox, crop):
        """
        Aggiorna la bounding box e il crop della traccia, resettando il contatore di frame saltati.
        
        :param bbox: Nuova bounding box della traccia.
        :param crop: Nuovo crop della traccia.
        """
        self.bbox = bbox
        self.crop = crop
        self.skipped_frames = 0
        logging.info(f"Track {self.track_id} updated with new bbox and crop.")

    def increment_skipped_frames(self):
        """
        Incrementa il contatore di frame saltati.
        """
        self.skipped_frames += 1
        logging.info(f"Track {self.track_id} skipped frame incremented to {self.skipped_frames}.")

    def is_lost(self):
        """
        Verifica se la traccia è persa (se il numero di frame saltati supera la soglia massima).
        
        :return: True se la traccia è persa, False altrimenti.
        """
        return self.skipped_frames > self.max_frames_to_skip
