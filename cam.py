import os
import cv2

# ── Config ────────────────────────────────────────────────────
DATA_DIR    = './data'
IMAGES_PER_CLASS = 300   # number of images to collect per sign

# ─────────────────────────────────────────────────────────────

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("="*50)
print("  ASL Data Collector")
print("="*50)
print(f"  Images per class : {IMAGES_PER_CLASS}")
print(f"  Save directory   : {DATA_DIR}")
print("="*50)

label = input("Enter the label/class name for this sign (e.g. A, B, HELLO): ").strip()

if label == "":
    print("No label entered. Exiting.")
    cap.release()
    exit()

class_dir = os.path.join(DATA_DIR, label)
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

print(f"\nCollecting {IMAGES_PER_CLASS} images for class: '{label}'")
print("Get your hand ready, then press 'S' to start capturing.")
print("Press 'Q' to quit early.\n")

# ── Wait for user to press S ──────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"Class: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, "Press S to Start | Q to Quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("ASL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        break
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ── Capture images ────────────────────────────────────────────
count = 0
while count < IMAGES_PER_CLASS:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    # Save the frame
    img_path = os.path.join(class_dir, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1

    # Show progress on screen
    progress = int((count / IMAGES_PER_CLASS) * 100)
    bar_w    = int((frame.shape[1] - 20) * (count / IMAGES_PER_CLASS))
    cv2.rectangle(frame, (10, frame.shape[0]-25),
                  (frame.shape[1]-10, frame.shape[0]-10), (60,60,60), -1)
    cv2.rectangle(frame, (10, frame.shape[0]-25),
                  (10+bar_w, frame.shape[0]-10), (0,210,0), -1)

    cv2.putText(frame, f"Class: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Capturing: {count}/{IMAGES_PER_CLASS} ({progress}%)", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit early", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.imshow("ASL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"Stopped early. Saved {count} images.")
        break

print(f"\nDone! Saved {count} images to: {class_dir}")
print("You can now run train.ipynb to retrain the model.")

cap.release()
cv2.destroyAllWindows()
