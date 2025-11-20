# pip install opencv-python
import cv2

def video_to_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error: Could not open video')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f'frame_{saved_frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f'Total frames saved: {saved_frame_count}')

video_to_frame('video.mp4')
