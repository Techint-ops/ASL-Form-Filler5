from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import pickle, cv2, numpy as np

app = FastAPI()

# Allow requests from any website (needed for browser integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model_data = pickle.load(open('classifier/classify_letter_model.p', 'rb'))
model = model_data['model']

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

@app.get("/")
def root():
    return {"status": "ASL API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = mp_hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return {"letter": None}

    lm = result.multi_hand_landmarks[0].landmark
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    data = []
    for l in lm:
        data.append(l.x - min(xs))
        data.append(l.y - min(ys))

    prediction = model.predict([data])[0]
    return {"letter": str(prediction)}
