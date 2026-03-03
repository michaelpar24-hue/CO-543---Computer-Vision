from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import time
import random



app = FastAPI(title="Red Light Green Light Project")

#----------------------Parameters Value----------------------#

STATE_START   = "START"
STATE_GREEN   = "GREEN"
STATE_WARNING = "WARNING"
STATE_RED     = "RED"
STATE_DEAD    = "DEAD"

FRAME_WIDTH = 640

GREEN_MOVE_THRESHOLD = 0.04
RED_MOVE_THRESHOLD   = 0.055

RED_GRACE_MS     = 650
IDLE_WARNING_MS  = 1800
IDLE_DEATH_MS    = 3600

GREEN_MIN_MS = 2600
GREEN_MAX_MS = 4200
RED_MIN_MS   = 1700
RED_MAX_MS   = 2900


state = STATE_START
state_start_time = time.time()
idle_start_time = None

prev_gray = None
level = 0

green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS)
red_duration   = random.uniform(RED_MIN_MS, RED_MAX_MS)

#-------------------Camera Section-------------------------#

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("❌ Webcam could not be opened")

#------------------------def functions------------------------#

def current_millis() -> int:
    return int(time.time() * 1000)

def randomize_durations() -> None:
    global green_duration, red_duration
    green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS)
    red_duration   = random.uniform(RED_MIN_MS, RED_MAX_MS)

def resize_frame(frame: np.ndarray) -> np.ndarray: #because we have a fixed width we need a scale ratio for the height
    h, w = frame.shape[:2]
    scale = FRAME_WIDTH / w
    return cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))#yung frame width ay fix so yung height multiplied by scale to get it, proportionality

def calculate_motion(prev: np.ndarray, current: np.ndarray) -> float:#ito yung prevoius fram vs current to get the absdiff
    diff = cv2.absdiff(prev, current)
    return np.mean(diff) / 255.0

def generate_frames():
    global state, prev_gray, state_start_time, idle_start_time, level


    #-----------------------Main-------------------------------#
    while True:
        success, frame = camera.read()
        if not success:
            break

        # ---------------- Frame Preprocessing ---------------- #
        frame = resize_frame(frame)#fucntion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert to grayscale
        gray = cv2.GaussianBlur(gray, (5, 5), 0)#reduce noise, it smooth the image by averaging nearby pixels

        motion_score = 0.0
        if prev_gray is not None:
            motion_score = calculate_motion(prev_gray, gray)

        prev_gray = gray.copy()

        now = current_millis()
        elapsed = now - int(state_start_time * 1000)

        # ---------------- Game State Machine ---------------- #
        if state == STATE_GREEN:
            moving = motion_score >= GREEN_MOVE_THRESHOLD
            if not moving:
                idle_start_time = idle_start_time or now
                idle_elapsed = now - idle_start_time
                if idle_elapsed > IDLE_DEATH_MS:
                    state = STATE_DEAD
                elif idle_elapsed > IDLE_WARNING_MS:
                    state = STATE_WARNING
            else:
                idle_start_time = None
            if elapsed > green_duration:
                level += 1
                state = STATE_RED
                state_start_time = time.time()

        elif state == STATE_WARNING:
            cv2.putText(frame, "MAKE A MOVE!", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
            if motion_score >= GREEN_MOVE_THRESHOLD:
                
                state = STATE_GREEN
                idle_start_time = None
            elif now - idle_start_time > IDLE_DEATH_MS:
                state = STATE_DEAD

        elif state == STATE_RED:
            if elapsed > RED_GRACE_MS and motion_score > RED_MOVE_THRESHOLD:
                state = STATE_DEAD
            if elapsed > red_duration:
                state = STATE_GREEN
                state_start_time = time.time()
                idle_start_time = None
                randomize_durations()

        # ---------------- UI Overlay ---------------- #
        color = (0, 255, 0)
        if state == STATE_RED:
            color = (0, 0, 255)
        elif state == STATE_WARNING:
            color = (0, 165, 255)
        elif state == STATE_DEAD:
            color = (0, 0, 139)

        cv2.putText(frame, f"STATE: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Motion: {motion_score:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Level: {level}", (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if state == STATE_DEAD:
            cv2.putText(frame, "YOU DIED!", (180, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # ---------------- Encode for Streaming ---------------- #
        _, buffer = cv2.imencode(".jpg", frame)#Converts an OpenCV image into a JPEG byte stream and sends it as one frame of a multipart HTTP video stream (MJPEG).
        frame_bytes = buffer.tobytes()#Converts NumPy array → raw byte string
        yield ( #sends frames one at a time
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes + b"\r\n"
        )


#---------------------API Section-----------------------------#


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/start")
def start_game():
    global state, state_start_time, idle_start_time, level
    state = STATE_GREEN
    level = 0
    idle_start_time = None
    state_start_time = time.time()
    randomize_durations()
    return {"state": state}

#@app.post("/stop")
#def stop_game():
    #global state
    #state = STATE_RED
    #return {"state": state}

@app.get("/", response_class=HTMLResponse) #FastAPI route decorator, "/" means home
def home():
    return """
    <html>
    <head>
        <title>Red Light Green Light Project</title>
        <style>
            body {
                background:#1f3b5b;
                color:white;
                text-align:center;
                font-family:Arial, Helvetica, sans-serif;
            }
            .buttons {
                display:flex;
                gap:16px;
                justify-content:center;
                margin-bottom:20px;
            }
            button {
                padding:14px 28px;
                font-size:14px;
                font-weight:bold;
                border:none;
                border-radius:10px;
                cursor:pointer;
                color:white;
                letter-spacing:0.5px;
                transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
            }
            button.start {
                background: linear-gradient(135deg, #22c55e, #16a34a);
                box-shadow: 0 0 14px rgba(34, 197, 94, 0.45);
            }
            button.stop {
                background: linear-gradient(135deg, #ef4444, #b91c1c);
                box-shadow: 0 0 14px rgba(239, 68, 68, 0.45);
            }
            button:hover {
                transform:translateY(-2px);
                filter:brightness(1.1);
            }
            button:active {
                transform:translateY(0);
                box-shadow:0 0 6px rgba(255,255,255,0.3);
            }
        </style>
    </head>
    <body>
        <h2>Red Light Green Light Project</h2>
        <div class="buttons">
            <button class="start" onclick="fetch('/start',{method:'POST'})">START</button>
            <!-- <button class="stop" onclick="fetch('/stop',{method:'POST'})">STOP</button> -->
        </div>
        <img src="/video" width="640">
    </body>
    </html>
    """


@app.on_event("shutdown")
def shutdown():
    camera.release()
