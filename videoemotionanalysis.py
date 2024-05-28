import cv2
import mediapipe as mp
import math
import os

def analyze_posture(video_path):
    """
    Analyzes the posture in a video.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing analysis results including overall posture
              and the percentage of good posture frames.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    good_posture_count = 0
    bad_posture_count = 0
    total_frames = 0
    incomplete_landmark_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            head_tilt_angle = abs(math.degrees(math.atan2(
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y - landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x - landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER].x)))
            shoulder_slope_angle = abs(math.degrees(math.atan2(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)))

            head_tilt_angle = min(head_tilt_angle, 180 - head_tilt_angle)
            shoulder_slope_angle = min(shoulder_slope_angle, 180 - shoulder_slope_angle)

            if head_tilt_angle > 15 or shoulder_slope_angle > 10:
                bad_posture_count += 1
            else:
                good_posture_count += 1
        else:
            incomplete_landmark_frames += 1

    cap.release()

    evaluated_frames = total_frames - incomplete_landmark_frames
    good_posture_percentage = (good_posture_count / evaluated_frames) * 100 if evaluated_frames > 0 else 0
    overall_posture = 'good' if good_posture_count > bad_posture_count else 'bad'

    return {
        "video_name": os.path.basename(video_path),
        "overall_posture": overall_posture,
        "good_posture_percentage": good_posture_percentage
    }

