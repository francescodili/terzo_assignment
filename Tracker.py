from keras.applications.resnet50 import ResNet50
from main import match_detections_to_tracks
import Track
import logging

logging.basicConfig(level=logging.INFO)

class Tracker:
    def __init__(self, similarity_threshold):
        self.tracks = {}
        self.next_id = 0
        self.resnet = ResNet50(weights='imagenet', include_top=False)
        self.lost_tracks = {}
        self.removed_track_ids = set()  # Set per tenere traccia degli ID rimossi
        self.similarity_threshold = similarity_threshold

    def add_track(self, bbox, crop):
        """
        Aggiunge una nuova traccia con la bounding box e il crop forniti.
        
        :param bbox: Bounding box della nuova traccia.
        :param crop: Crop della nuova traccia.
        """
        while self.next_id in self.removed_track_ids:
            self.next_id += 1

        self.tracks[self.next_id] = Track(self.next_id, bbox, crop)
        self.lost_tracks[self.next_id] = 0
        logging.info(f"Added new track with ID {self.next_id}.")
        self.next_id += 1

    def update_tracks(self, detected_bboxes, crops):
        """
        Aggiorna le tracce con le bounding box e i crop rilevati.
        
        :param detected_bboxes: Bounding box rilevate.
        :param crops: Crop rilevati.
        """
        current_tracks = list(self.tracks.values())
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(
            detected_bboxes, crops, current_tracks, self.resnet, similarity_threshold=self.similarity_threshold)

        for match in matches:
            track_idx, detection_idx = match
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].update(detected_bboxes[detection_idx], crops[detection_idx])
                self.lost_tracks[track_id] = 0
            else:
                logging.warning(f"track_idx {track_idx} out of range for current_tracks")

        for track_idx in unmatched_tracks:
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].increment_skipped_frames()  # Incrementa i frame saltati
                if self.tracks[track_id].is_lost():
                    del self.tracks[track_id]
                    del self.lost_tracks[track_id]
                    self.removed_track_ids.add(track_id)
                    logging.info(f"Removed track with ID {track_id} due to too many skipped frames.")
            else:
                logging.warning(f"track_idx {track_idx} out of range for current_tracks")

        for detection_idx in unmatched_detections:
            self.add_track(detected_bboxes[detection_idx], crops[detection_idx])

    def get_active_tracks(self):
        """
        Restituisce le tracce attive (non perse).
        
        :return: Lista delle tracce attive.
        """
        return [track for track in self.tracks.values() if not track.is_lost()]
