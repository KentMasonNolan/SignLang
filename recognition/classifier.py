# recognition/classifier.py
def classify_simple(results):

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0].landmark

    # Utility: check if finger is extended
    def finger_extended(tip, pip):
        return hand_landmarks[tip].y < hand_landmarks[pip].y

    # Finger landmark indices
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18
    THUMB_TIP, THUMB_IP = 4, 3

    # A = fingers curled, thumb alongside
    if (not finger_extended(INDEX_TIP, INDEX_PIP) and
        not finger_extended(MIDDLE_TIP, MIDDLE_PIP) and
        not finger_extended(RING_TIP, RING_PIP) and
        not finger_extended(PINKY_TIP, PINKY_PIP)):
        return "A"

    # B = fingers extended, thumb across palm
    if (finger_extended(INDEX_TIP, INDEX_PIP) and
        finger_extended(MIDDLE_TIP, MIDDLE_PIP) and
        finger_extended(RING_TIP, RING_PIP) and
        finger_extended(PINKY_TIP, PINKY_PIP)):
        return "B"

    # C = fingertips curve towards thumb (approximate check)
    # Distance between index tip and thumb tip should be smaller than in "B"
    thumb_tip = hand_landmarks[THUMB_TIP]
    index_tip = hand_landmarks[INDEX_TIP]
    distance = abs(index_tip.x - thumb_tip.x)
    if distance < 0.15:  # tweak threshold experimentally
        return "C"

    return None
