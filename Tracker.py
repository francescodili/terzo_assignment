from keras.applications.resnet50 import ResNet50
from main import match_detections_to_tracks
import Track

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.resnet = ResNet50(weights='imagenet', include_top=False)

    def add_track(self, bbox, crop):
        self.tracks[self.next_id] = Track(self.next_id, bbox, crop)
        self.next_id += 1

    def update_tracks(self, detected_bboxes, crops):
        current_tracks = list(self.tracks.values())
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(detected_bboxes, crops, current_tracks, self.resnet)

        print("Current tracks before update:", [(track.track_id, track.bbox) for track in current_tracks])
        print("Matches:", matches)
        print("Unmatched detections:", unmatched_detections)
        print("Unmatched tracks:", unmatched_tracks)

        for match in matches:
            track_idx, detection_idx = match
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].update(detected_bboxes[detection_idx], crops[detection_idx])
            else:
                print(f"Warning: track_idx {track_idx} out of range for current_tracks")

        for track_idx in unmatched_tracks:
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].increment_skipped_frames()
                if self.tracks[track_id].is_lost():
                    del self.tracks[track_id]
            else:
                print(f"Warning: track_idx {track_idx} out of range for current_tracks")

        for detection_idx in unmatched_detections:
            self.add_track(detected_bboxes[detection_idx], crops[detection_idx])

        print("Tracks after update:", [(track.track_id, track.bbox) for track in self.tracks.values()])

    def get_active_tracks(self):
        return [track for track in self.tracks.values() if not track.is_lost()]