import cv2
import os

# Change this to rock / paper / scissors before running
label = "paper"

save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 300  # number of images per class

print("Press SPACE to capture image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (200, 100), (450, 350), (0, 255, 0), 2)
    roi = frame[100:350, 200:450]

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, roi)
        count += 1
        print(f"Captured {count}")

    if key == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()