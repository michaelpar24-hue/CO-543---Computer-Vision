from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import random
from threading import Lock

#uvicorn backend:app --reload

app = FastAPI()

# Allow browser to access
app.add_middleware( #Browser → Middleware → FastAPI endpoint → Middleware → Browser
    CORSMiddleware,#A browser security rule that blocks requests from different origins unless allowed.
    allow_origins=["*"], #Allow ANY website to access this API
    allow_methods=["*"],#This allows all HTTP methods such as get, post, put, delete, patch, option, 
    allow_headers=["*"],
)

# ---------------- RLGL Parameters ----------------
__Start = "START"
GREEN = "GREEN"
WARNING = "WARNING"
RED = "RED"
DEAD = "DEAD"

GREEN_MOVE_THRESHOLD = 0.04
RED_MOVE_THRESHOLD = 0.055

RED_GRACE_MS = 650
IDLE_WARNING_MS = 1800
IDLE_DEATH_MS = 3600

GREEN_MIN_MS = 2600
GREEN_MAX_MS = 4200
RED_MIN_MS = 1700
RED_MAX_MS = 2900

FRAME_WIDTH = 640

# ---------------- RLGL State ----------------
state_lock = Lock()
rlgl_state = __Start
state_start_time = time.time()
idle_start_time = None
level_up = 0
motion_score = 0.0
green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS)
red_duration = random.uniform(RED_MIN_MS, RED_MAX_MS)
prev_gray = None

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ---------------- Define Functions ----------------
def current_millis():
    return int(time.time() * 1000)

def randomize_durations():
    global green_duration, red_duration
    green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS)
    red_duration = random.uniform(RED_MIN_MS, RED_MAX_MS)

def process_frame():
    global prev_gray, motion_score, rlgl_state, state_start_time, idle_start_time, level_up

    ret, frame = cap.read()
    if not ret:
        return None

    # Resize frame
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / w
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))

    # Grayscale & blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Motion score
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion_score = np.mean(diff) / 255.0
    prev_gray = gray.copy()

    now = current_millis()
    elapsed = now - int(state_start_time * 1000)

    # ---------------- RLGL Logic ----------------
    if rlgl_state == GREEN:
        moving = motion_score >= GREEN_MOVE_THRESHOLD
        if not moving:
            idle_start_time = idle_start_time or now
            idle_elapsed = now - idle_start_time
            if idle_elapsed > IDLE_DEATH_MS:
                rlgl_state = DEAD
            elif idle_elapsed > IDLE_WARNING_MS:
                rlgl_state = WARNING
        else:
            idle_start_time = None

        if elapsed > green_duration:
            level_up += 1
            rlgl_state = RED
            state_start_time = time.time()

    elif rlgl_state == WARNING:
        if motion_score >= GREEN_MOVE_THRESHOLD:
            rlgl_state = GREEN
            idle_start_time = None
        elif now - idle_start_time > IDLE_DEATH_MS:
            rlgl_state = DEAD

    elif rlgl_state == RED:
        if elapsed > RED_GRACE_MS and motion_score > RED_MOVE_THRESHOLD:
            rlgl_state = DEAD
        if elapsed > red_duration:
            rlgl_state = GREEN
            state_start_time = time.time()
            idle_start_time = None
            randomize_durations()

    elif rlgl_state == __Start:
        # Waiting for Start
        pass

    return frame

# ---------------- Video Streaming ----------------
def generate_frames():
    while True:
        frame = process_frame()
        if frame is None:
            continue

        # Draw info on frame
        color = (0, 255, 0)
        if rlgl_state == RED:
            color = (0, 0, 255)
        elif rlgl_state == WARNING:
            color = (0, 165, 255)
        elif rlgl_state == DEAD:
            color = (0, 0, 139)

        cv2.putText(frame, f"STATE: {rlgl_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Motion Score: {motion_score:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Level: {level_up}", (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if rlgl_state == DEAD:
            h, w = frame.shape[:2]

            text = "X"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 10     # very large like Header 1
            thickness = 25

            # get text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2

            cv2.putText(frame, text, (text_x, text_y),
                        font, font_scale, (0, 0, 255), thickness)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------- API Endpoints ----------------
@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/status")
def get_status():
    with state_lock:
        return JSONResponse({
            "state": rlgl_state,
            "motion_score": round(motion_score,3),
            "level": level_up
        })

@app.post("/start")
def start_rlgl():
    global rlgl_state, state_start_time, idle_start_time, level_up
    with state_lock:
        rlgl_state = GREEN
        state_start_time = time.time()
        idle_start_time = None
        level_up = 0
        randomize_durations()
    return {"state": rlgl_state}

@app.post("/stop")
def stop_rlgl():
    #global rlgl_state
    #with state_lock:
       # rlgl_state = RED
    #return {"state": rlgl_state}
    global rlgl_state, state_start_time, idle_start_time, level_up
    with state_lock:
        rlgl_state = GREEN
        state_start_time = time.time()
        idle_start_time = None
        level_up = 0
        randomize_durations()
    return {"state": rlgl_state}


@app.post("/restart")
def restart_rlgl():
    global rlgl_state, state_start_time, idle_start_time, level_up
    with state_lock:
        rlgl_state = GREEN
        state_start_time = time.time()
        idle_start_time = None
        level_up = 0
        randomize_durations()
    return {"state": rlgl_state}
