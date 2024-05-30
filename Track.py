

class Track:
    def __init__(self, track_id, bbox, crop, max_frames_to_skip=5):
        self.track_id = track_id
        self.bbox = bbox
        self.crop = crop
        self.max_frames_to_skip = max_frames_to_skip
        self.skipped_frames = 0

    def update(self, bbox, crop):
        self.bbox = bbox
        self.crop = crop
        self.skipped_frames = 0

    def increment_skipped_frames(self):
        self.skipped_frames += 1

    def is_lost(self):
        return self.skipped_frames > self.max_frames_to_skip