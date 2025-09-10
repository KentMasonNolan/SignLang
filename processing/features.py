# processing/features.py
import numpy as np

def extract_features(results):

    if not results.multi_hand_landmarks:
        return None

    for hand_landmarks in results.multi_hand_landmarks:
        # Collect raw coords
        coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

        # Normalize relative to wrist (landmark 0)
        wrist = coords[0]
        coords -= wrist

        # Scale normalization (distance wrist→middle finger MCP = index 9)
        scale = np.linalg.norm(coords[9])
        if scale > 0:
            coords /= scale

        return coords.flatten().tolist()   # 63 normalized values
