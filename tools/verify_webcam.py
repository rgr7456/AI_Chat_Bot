"""
Simple webcam client to capture frames and send to the API /face/verify endpoint.
Usage:
  python tools/verify_webcam.py --url http://127.0.0.1:8000/face/verify

Press 'v' to capture the current frame and send it for verification.
Press 'q' to quit.

Requires: opencv-python, requests
Install: pip install opencv-python requests
"""
import argparse
import cv2
import requests
import time
import numpy as np


def send_frame(url, frame, timeout=10):
    # encode as jpeg
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        raise RuntimeError('Failed to encode frame')
    files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(url, files=files, timeout=timeout)
        return r
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/face/verify', help='Verify endpoint URL')
    parser.add_argument('--cam', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print("Press 'v' to verify current frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame')
            break

        display = frame.copy()
        cv2.putText(display, "Press v to verify, q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('webcam', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('v'):
            print('Sending frame to', args.url)
            start = time.time()
            resp = send_frame(args.url, frame)
            elapsed = time.time() - start
            if resp is None:
                print('Request failed (timeout or connection error)')
                continue
            try:
                print('Status code:', resp.status_code)
                print('Response JSON:', resp.json())
            except Exception as e:
                print('Failed to parse response:', e)
            print(f'Elapsed {elapsed:.2f}s')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
