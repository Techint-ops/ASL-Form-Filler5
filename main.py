import sys
import os
import cv2
import time
import argparse
import numpy as np
import mediapipe as mp

from autocorrect import Speller
from utils import load_model, save_gif, save_video
from utils import calc_landmark_list, draw_landmarks, draw_info_text

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang='en')

# Colors BGR Format
BLACK  = (0,   0,   0)
RED    = (0,   0,   255)
GREEN  = (0,   255, 0)
BLUE   = (255, 0,   0)
YELLOW = (0,   255, 255)
WHITE  = (255, 255, 255)
GRAY   = (60,  60,  60)
LGRAY  = (200, 200, 200)
PANEL  = (30,  30,  30)
ACCENT = (50,  160, 255)
DARK   = (20,  20,  20)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Constants
MAX_HANDS = 1
min_detection_confidence = 0.3
min_tracking_confidence  = 0.3

MODEL_PATH        = "./classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"
model_number_path = f"{MODEL_PATH}/classify_number_model.p"

# Form layout
FORM_W  = 480
CAM_W   = 800
CAM_H   = 600

# Form fields
FIELDS = ["First Name", "Last Name", "Age", "Phone", "Email", "Address", "City"]
form_data    = {f: "" for f in FIELDS}
active_field = 0

# Confirmation timing (frames gesture must be held)
CONFIRM_FRAMES = 40
confirm_counter = 0
last_gesture    = ""
flash_counter   = 0
flash_char      = ""


def check_model_files():
    """Verify that model files exist before starting"""
    missing = []
    
    if not os.path.exists(model_letter_path):
        missing.append(model_letter_path)
    if not os.path.exists(model_number_path):
        missing.append(model_number_path)
    
    if missing:
        print("❌ ERROR: Model files not found:")
        for path in missing:
            print(f"   - {path}")
        print("\nPlease ensure model files are in the ./classifier directory")
        return False
    
    print("✅ Model files found")
    return True


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',      type=str, default=None,  help='Video Path/0 for Webcam')
    parser.add_argument('-a', '--autocorrect', action='store_true',      help='Autocorrect Misspelled Word')
    parser.add_argument('-g', '--gif',         action='store_true',      help='Save GIF Result')
    parser.add_argument('-v', '--video',       action='store_true',      help='Save Video Result')
    parser.add_argument('-t', '--timing',      type=int, default=8,      help='Timing Threshold')
    parser.add_argument('-wi','--width',       type=int, default=800,    help='Webcam Width')
    parser.add_argument('-he','--height',      type=int, default=600,    help='Webcam Height')
    parser.add_argument('-f', '--fps',         type=int, default=30,     help='Webcam FPS')
    opt = parser.parse_args()
    return opt


def get_output(idx):
    global _output, output, autocorrect, TIMING
    key = []
    for i in range(len(_output[idx])):
        character = _output[idx][i]
        counts = _output[idx].count(character)
        if (character not in key) or (character != key[-1]):
            if counts > TIMING:
                key.append(character)

    text = ""
    for character in key:
        if character == "?":
            continue
        text += str(character).lower()

    text = spell(text) if autocorrect else text

    if text != "":
        _output[idx] = []
        output.append(text.title())
    return None


def recognize_gesture(image, results, numberMode, letter_model, number_model):
    """
    FIXED: Models are now passed as parameters instead of loading every frame
    """
    global current_hand, output, _output

    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness     = results.multi_handedness

    _gesture   = []
    data_aux   = []
    detected_gesture = ""

    isIncreased = False
    isDecreased = False

    if current_hand != 0:
        if results.multi_hand_landmarks is None:
            isDecreased = True
        else:
            if len(multi_hand_landmarks) > current_hand:
                isIncreased = True
            elif len(multi_hand_landmarks) < current_hand:
                isDecreased = True

    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        for idx in reversed(range(len(multi_hand_landmarks))):
            current_select_hand = multi_hand_landmarks[idx]
            handness = multi_handedness[idx].classification[0].label

            mp_drawing.draw_landmarks(
                image,
                current_select_hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            landmark_list = calc_landmark_list(image, current_select_hand)
            image = draw_landmarks(image, landmark_list)

            x_values = [lm.x for lm in current_select_hand.landmark]
            y_values = [lm.y for lm in current_select_hand.landmark]

            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)

            cv2.putText(image, f"{handness} Hand", (min_x - 10, max_y + 30), FONT, 0.5, GREEN, 2)

            if handness == 'Left':
                x_values = list(map(lambda x: 1 - x, x_values))
                min_x -= 10

            for i in range(len(current_select_hand.landmark)):
                data_aux.append(x_values[i] - min(x_values))
                data_aux.append(y_values[i] - min(y_values))

            # FIXED: Better error handling with detailed messages
            try:
                if not numberMode:
                    prediction = letter_model.predict([np.asarray(data_aux)])
                    gesture = str(prediction[0]).title()
                    gesture = gesture if gesture != 'Unknown_Letter' else '?'
                    print(f"[DEBUG] Letter prediction: {gesture} (raw: {prediction[0]})")
                else:
                    prediction = number_model.predict([np.asarray(data_aux)])
                    gesture = str(prediction[0]).title()
                    gesture = gesture if gesture != 'Unknown_Number' else '?'
                    print(f"[DEBUG] Number prediction: {gesture} (raw: {prediction[0]})")
            except Exception as e:
                gesture = '?'
                print(f"❌ Prediction error: {e}")
                print(f"   Data aux length: {len(data_aux)}")
                print(f"   Expected: 42 (21 landmarks x 2 coordinates)")
                import traceback
                traceback.print_exc()

            cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), BLACK, 4)
            image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

            _gesture.append(gesture)
            detected_gesture = gesture

    if isDecreased == True:
        if current_hand == 1:
            get_output(0)
    else:
        if results.multi_hand_landmarks is not None:
            _output[0].append(_gesture[0])

    if results.multi_hand_landmarks:
        current_hand = len(multi_hand_landmarks)
    else:
        current_hand = 0

    return image, detected_gesture


def draw_form_panel(win_h):
    """Draw the form panel and return it as an image."""
    panel = np.zeros((win_h, FORM_W, 3), dtype=np.uint8)
    panel[:] = PANEL

    # Title bar
    cv2.rectangle(panel, (0, 0), (FORM_W, 52), DARK, -1)
    cv2.putText(panel, "ASL Form Filler", (12, 35), FONT_BOLD, 0.9, ACCENT, 2)
    cv2.putText(panel, "Hold sign to confirm letter", (12, 68), FONT, 0.38, LGRAY, 1)

    y = 90
    for i, field in enumerate(FIELDS):
        is_active = (i == active_field)
        value     = form_data[field]

        cv2.putText(panel, field.upper(), (12, y),
                    FONT, 0.40, ACCENT if is_active else LGRAY, 1)
        y += 20

        box_h = 34
        cv2.rectangle(panel, (8, y), (FORM_W-8, y+box_h),
                      ACCENT if is_active else GRAY,
                      2 if is_active else 1)
        if is_active:
            cv2.rectangle(panel, (10, y+2), (FORM_W-10, y+box_h-2), (45,45,45), -1)

        cursor = "|" if is_active and int(time.time()*2) % 2 == 0 else ""
        cv2.putText(panel, value + cursor, (15, y+23), FONT, 0.58, WHITE, 1)
        y += box_h + 10

    # Flash confirmed character
    if flash_counter > 0:
        cv2.rectangle(panel, (8, win_h-100), (FORM_W-8, win_h-65), (0,80,0), -1)
        cv2.putText(panel, f"  Added: '{flash_char}'", (12, win_h-73),
                    FONT_BOLD, 0.7, GREEN, 2)

    # Bottom bar
    cv2.rectangle(panel, (0, win_h-58), (FORM_W, win_h), DARK, -1)
    cv2.putText(panel, "TAB:next field  Q:prev  SPC:space  BKSP:del",
                (8, win_h-38), FONT, 0.34, LGRAY, 1)
    cv2.putText(panel, "ENTER:save form  C:clear  M:mode  ESC:quit",
                (8, win_h-18), FONT, 0.34, LGRAY, 1)

    return panel


def save_form():
    print("\n" + "="*40)
    print("  FORM SUBMITTED")
    print("="*40)
    for field in FIELDS:
        print(f"  {field:12}: {form_data[field]}")
    print("="*40)
    with open("form_output.txt", "w") as f:
        f.write("ASL FORM SUBMISSION\n" + "="*40 + "\n")
        for field in FIELDS:
            f.write(f"{field}: {form_data[field]}\n")
    print("✅ Saved to form_output.txt")


if __name__ == '__main__':
    # FIXED: Check model files first
    if not check_model_files():
        print("\n⚠️  Cannot start without model files")
        sys.exit(1)
    
    opt = parse_opt()
    saveGIF    = opt.gif
    saveVDO    = opt.video
    source     = opt.source

    global TIMING, autocorrect
    TIMING      = opt.timing
    autocorrect = opt.autocorrect
    print(f"⚙️  Timing Threshold: {TIMING} frames")
    print(f"⚙️  Autocorrect: {autocorrect}")

    if source is None or source.isnumeric():
        video_path = 0
    else:
        video_path = source

    fps           = opt.fps
    webcam_width  = opt.width
    webcam_height = opt.height

    _output      = [[], []]
    output       = []
    quitApp      = False
    frame_array  = []
    current_hand = 0
    numberMode   = False

    # FIXED: Load models ONCE before main loop
    print("📦 Loading models...")
    try:
        letter_model = load_model(model_letter_path)
        number_model = load_model(model_number_path)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Webcam setup ──────────────────────────────────────────
    if video_path == 0:
        capture = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,  webcam_width)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FPS, fps)

        # Ready screen
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            text = 'Ready? Press "R" | ESC to Quit'
            textsize = cv2.getTextSize(text, FONT, 1.0, 3)[0]
            textX = (frame.shape[1] - textsize[0]) // 2
            textY = (frame.shape[0] + textsize[1]) // 2
            cv2.putText(frame, text, (textX, textY), FONT, 1.0, GREEN, 3, cv2.LINE_AA)

            form_panel = draw_form_panel(frame.shape[0])
            combined   = np.hstack([frame, form_panel])
            cv2.imshow('ASL Form Filler', combined)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('r'):
                break
            if key == 27:
                quitApp = True
                break

        cv2.destroyAllWindows()
        if quitApp:
            capture.release()
            quit()

    else:
        capture = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)

    if not capture.isOpened():
        print("❌ Error: Could not open webcam")
        exit()

    print("✅ Camera opened successfully")
    print("🚀 Starting gesture recognition...")

    # ── Main loop ─────────────────────────────────────────────
    with mp_hands.Hands(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        max_num_hands=MAX_HANDS
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                if video_path == 0:
                    continue
                else:
                    print("Video ends.")
                    break

            if video_path == 0:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_width  = int(capture.get(3))
                frame_height = int(capture.get(4))

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Always draw skeleton
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Hand detection status
            if results.multi_hand_landmarks:
                cv2.putText(image, f"Hand Detected", (10, image.shape[0]-20),
                            FONT, 0.6, GREEN, 2)
            else:
                cv2.putText(image, "No Hand Detected", (10, image.shape[0]-20),
                            FONT, 0.6, RED, 2)

            # Gesture recognition - FIXED: Pass loaded models as parameters
            detected_gesture = ""
            try:
                image, detected_gesture = recognize_gesture(
                    image, results, numberMode, letter_model, number_model)
            except Exception as error:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f"❌ ERROR: {error}, line {exc_tb.tb_lineno}")
                import traceback
                traceback.print_exc()

            # ── Form confirmation logic ────────────────────────
            if detected_gesture and detected_gesture != '?':
                if detected_gesture == last_gesture:
                    confirm_counter += 1
                else:
                    confirm_counter = 0
                    last_gesture = detected_gesture

                # Progress bar
                bar_w = int((image.shape[1] - 20) * (confirm_counter / CONFIRM_FRAMES))
                cv2.rectangle(image, (10, image.shape[0]-12),
                              (image.shape[1]-10, image.shape[0]-4), GRAY, -1)
                bar_col = GREEN if confirm_counter >= CONFIRM_FRAMES else ACCENT
                cv2.rectangle(image, (10, image.shape[0]-12),
                              (10+bar_w, image.shape[0]-4), bar_col, -1)
                pct = int((confirm_counter / CONFIRM_FRAMES) * 100)
                cv2.putText(image, f"Hold to confirm: {pct}%",
                            (10, image.shape[0]-16), FONT, 0.38, WHITE, 1)

                # Confirm into form field
                if confirm_counter >= CONFIRM_FRAMES:
                    field_name = FIELDS[active_field]
                    if len(detected_gesture) > 1:
                        form_data[field_name] += detected_gesture + " "
                    else:
                        form_data[field_name] += detected_gesture.lower()
                    flash_char      = detected_gesture
                    flash_counter   = 25
                    confirm_counter = 0
                    last_gesture    = ""
                    print(f"✅ Added '{detected_gesture}' to {field_name}")
            else:
                confirm_counter = 0
                last_gesture    = ""

            if flash_counter > 0:
                flash_counter -= 1

            # Show output text (top left — original behaviour)
            output_text = str(output)
            output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 0), (10+output_size[0], 10+output_size[1]), YELLOW, -1)
            cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

            mode_text = f"Number: {numberMode}"
            mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 45), (10+mode_size[0], 55+mode_size[1]), YELLOW, -1)
            cv2.putText(image, mode_text, (10, 60), FONT, 0.5, BLACK, 2)

            hint = "ESC:Quit | M:Mode | C:Clear | BKSP:Delete | S:Save | TAB:NextField"
            cv2.putText(image, hint, (10, image.shape[0]-35), FONT, 0.35, WHITE, 1)

            # Build combined window
            form_panel = draw_form_panel(image.shape[0])
            combined   = np.hstack([image, form_panel])
            cv2.imshow('ASL Form Filler', combined)
            frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(5) & 0xFF

            if key == 27:       # ESC
                print("ESC pressed, quitting...")
                break
            if key == 9:        # TAB — next field
                active_field = (active_field + 1) % len(FIELDS)
                confirm_counter = 0; last_gesture = ""
            if key == ord('q'): # Q — previous field
                active_field = (active_field - 1) % len(FIELDS)
                confirm_counter = 0; last_gesture = ""
            if key == 32:       # SPACE
                form_data[FIELDS[active_field]] += " "
            if key == 8:        # BACKSPACE
                field_name = FIELDS[active_field]
                if form_data[field_name]:
                    form_data[field_name] = form_data[field_name][:-1]
                elif output:
                    output.pop()
            if key == ord('c'): # C — clear field
                form_data[FIELDS[active_field]] = ""
                output.clear()
            if key == 13:       # ENTER — save form
                save_form()
                msg = combined.copy()
                cv2.rectangle(msg, (80, 220), (combined.shape[1]-80, 290), DARK, -1)
                cv2.rectangle(msg, (80, 220), (combined.shape[1]-80, 290), GREEN, 2)
                cv2.putText(msg, "Form saved to form_output.txt!",
                            (110, 263), FONT_BOLD, 0.85, GREEN, 2)
                cv2.imshow('ASL Form Filler', msg)
                cv2.waitKey(2000)
            if key == ord('s'): # S — save GIF/video
                saveGIF = True
                saveVDO = True
                break
            if key == ord('m'): # M — toggle mode
                numberMode = not numberMode
                print(f"Mode: {'Number' if numberMode else 'Letter'}")
            # Number keys 1-7 jump to field
            for i, k in enumerate([ord('1'),ord('2'),ord('3'),ord('4'),
                                    ord('5'),ord('6'),ord('7')]):
                if key == k and i < len(FIELDS):
                    active_field = i
                    confirm_counter = 0; last_gesture = ""

    cv2.destroyAllWindows()
    capture.release()

    print(f"\n📝 Gesture Recognition:\n{' '.join(output)}")

    if saveGIF:
        print("💾 Saving GIF Result..")
        save_gif(frame_array, fps=fps, output_dir="./assets/result_ASL.gif")

    if saveVDO:
        print("💾 Saving Video Result..")
        width  = webcam_width  if video_path == 0 else frame_width
        height = webcam_height if video_path == 0 else frame_height
        save_video(frame_array, fps=fps, width=width, height=height,
                   output_dir="./assets/result_ASL.mp4")