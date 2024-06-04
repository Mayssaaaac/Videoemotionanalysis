import cv2
from deepface import DeepFace

def detect_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    emotion_count = {}
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if analysis and isinstance(analysis, list):
                for result in analysis:
                    if 'dominant_emotion' in result:
                        dominant_emotion = result['dominant_emotion']
                        if dominant_emotion in emotion_count:
                            emotion_count[dominant_emotion] += 1
                        else:
                            emotion_count[dominant_emotion] = 1
        except Exception as e:
            print("Detection error:", e)

    cap.release()
    cv2.destroyAllWindows()

    if total_frames > 0:
        sorted_emotions = sorted(emotion_count.items(), key=lambda item: item[1], reverse=True)
        top_two_emotions = sorted_emotions[:2]
        
        print("Percentage of frames for the two most dominant emotions:")
        for emotion, count in top_two_emotions:
            percentage = (count / total_frames) * 100
            print(f"{emotion}: {percentage:.2f}%")
    else:
        print("No frames processed.")
