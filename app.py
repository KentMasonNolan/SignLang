# app.py
import cv2
from capture.webcam import get_camera_stream
from processing.landmarks import HandProcessor

def main():
    cap = get_camera_stream()
    processor = HandProcessor()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.process(frame_rgb)

        # Draw results
        frame_bgr = processor.draw_landmarks(frame, results)

        # Show window
        cv2.imshow("Sign Language Recognition", frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
