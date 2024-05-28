def start_track(tracks, detection):
    # Aggiungi una nuova traccia per una rilevazione
    tracks.append(detection)

def remove_track(tracks, track_id):
    # Rimuovi la traccia non più rilevata
    tracks = [track for track in tracks if track['id'] != track_id]
    return tracks

def update_tracks(tracks, detections):
    # Aggiorna le tracce esistenti con nuove rilevazioni
    # Implementa logica per associare rilevazioni a tracce esistenti
    pass
