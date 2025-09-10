# app.py
import cv2
from capture.webcam import get_camera_stream
from processing.landmarks import HandProcessor
from recognition.classifier import classify_simple

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

        frame_bgr = processor.draw_landmarks(frame, results)

        # Classify
        prediction = classify_simple(results)
        if prediction:
            print(prediction)  # console debug
            cv2.putText(frame_bgr, f"Detected: {prediction}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Show window AFTER drawing text
        cv2.imshow("Sign Language Recognition", frame_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
