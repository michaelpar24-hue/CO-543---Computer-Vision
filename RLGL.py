import cv2
import numpy as np
import time
import random
from ultralytics import YOLO

print("Loading AI Model...")
model = YOLO('yolov8n.pt')

#-----------------------------Duration funtion------------------------------------#

def current_millis():
    return int(time.time() * 1000)

def randomize_durations():
    global green_duration, red_duration #VERY IMPORTANT! this will modify the variables that exist outside this function
    green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS) #returns a random floating-point number
    red_duration = random.uniform(RED_MIN_MS, RED_MAX_MS) #returns a random floating-point number

#----------------------------Opening the camera--------------------------------------#

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam.")
    exit()

print("Camera started! Press 'e to exit")

#-----------------------------------Variables--------------------------------#
__Start = "START"
GREEN = "GREEN"
WARNING = "WARNING"
RED = "RED"
DEAD = "DEAD"

    #-----------------Suggested Baseline Parameters from PDF file--------------------#

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

    #-----------------------------Initial Values--------------------------------------#

prev_gray = None
state = __Start
state_start_time = time.time()
idle_start_time = None

green_duration = random.uniform(GREEN_MIN_MS, GREEN_MAX_MS)
red_duration = random.uniform(RED_MIN_MS, RED_MAX_MS)

    #------------------------------------Start of loop---------------------------------------------#

level_up = 0

while True:

    ret, frame = cap.read()
    # cv2.imshow("Webcam", frame)

        #----------------------Frame resizing----------------------------------------------------------#
        
    h, w = frame.shape[:2]
    # print("Ito yung value ng w at h", w, h)
    scale = FRAME_WIDTH / w
    #print("Scale naman ->", scale) # Si scale yung magdedefine kung ilan percent mag shrink yung image natin 
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale))) # para ito sa Height Auto adjustment #fixed kase si Frame width sa 640
    # print("Print natin sir frame->", frame) #array ang result ni frame
        
    #-----------------------Converting to grayscale(optional Blur)----------------------------------#
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #ito yung code for converting the colored image to grayscale, 3-channels to 1-channel # Remove color, keep brightness
    # print(f"gray 1 {gray}")
    gray = cv2.GaussianBlur(gray, (5, 5), 0) #this is the code for Blurring using Gaussian low-pass filter #Reduce noise
    # print(f"gray 2 {gray}")
        
    #-----------------------Motion dection section--------------------------------------------------#
        
    motion_score = 0.0 #Prepare motion measurement, initial value syempre
    try:

        if prev_gray is not None: #initial value natin ay None, we need to compare frame by frame to verify if there is a motion by looking at its brightness
            diff = cv2.absdiff(gray, prev_gray) #Computes the absolute pixel-wise difference between two frames #bright pixels means - big change or motion
            motion_score = np.mean(diff) / 255.0 #Nomalize to 0-1 parang sa numerical methods SVD Decom, dito makukuha yung avergae pixel change, Larger value = more overall motion
            # print("Section ito ng prev_gray, to check lang nman if dumaan dito!")
    except UnboundLocalError:
        #pass
        print("Hindi NOne yung value ng prev_gray natin!")
    
    prev_gray = gray.copy() #dito naka store yung prev frames to compare

    now = current_millis() # storing the current time in millisecond to this variable
    elapsed = now - int(state_start_time * 1000) # Measure time since state start
    #print(elapsed)
        
    #-------------------------Heart of the program----------------------#
    # print(motion_score)
    
    if state == GREEN:
        moving = motion_score >= GREEN_MOVE_THRESHOLD # green =  0.04

    # -------------idle tracking for green state----------------------------------#
        if not moving:
            idle_start_time = idle_start_time or now
            idle_elapsed = now - idle_start_time
            #print(f"Idle start time -> {idle_start_time}, NOW -> {now}")
            if idle_elapsed > IDLE_DEATH_MS: #time duration is 3600
                state = DEAD
            elif idle_elapsed > IDLE_WARNING_MS: #time duration is 1800
                state = WARNING
        else:
            idle_start_time = None

        # -----------------green timeout--------------------------------#
        if elapsed > green_duration:
            level_up += 1
            state = RED
            state_start_time = time.time()

        # ---------------------Warning condition---------------------------#
    elif state == WARNING:

        cv2.putText(frame, "You need to move!", (200, 200), #position (x=200, y=200)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if motion_score >= GREEN_MOVE_THRESHOLD:# green =  0.04
            state = GREEN
            idle_start_time = None
        elif now - idle_start_time > IDLE_DEATH_MS:#time duration is 3600
            state = DEAD

        # ---------------------Red state---------------------------#
    elif state == RED:
        if elapsed > RED_GRACE_MS and motion_score > RED_MOVE_THRESHOLD:
            state = DEAD

        if elapsed > red_duration: #red_duration = random.uniform(RED_MIN_MS, RED_MAX_MS)
            state = GREEN
            state_start_time = time.time()
            idle_start_time = None
            randomize_durations()

    # ---------------------Game Start---------------------------#
    elif state == __Start:
        cv2.putText(frame, "Press 's' to start or 'e' to exit", (175, 250), #position (x=200, y=200)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #---------------------------------Display Interface---------------------------------#
    color = (0, 255, 0)
    if state == RED:
        color = (0, 0, 255)
    elif state == WARNING:
        color = (0, 165, 255)
    elif state == DEAD:
        color = (0, 0, 139)

    cv2.putText(frame, f"STATE: {state}", (20, 40), #position (x=20, y=40)
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) #clean readable font, 1 → font scale (size), color → chosen based on state, 2 → text thickness

    cv2.putText(frame, f"Motion Score: {motion_score:.3f}", (20, 80), #:.3f → formatted to 3 decimal places, White text (255,255,255) so it’s always visible
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #clean readable font, 0.7 → font scale (size), color → chosen based on state, 2 → text thickness, Smaller font (0.7) than the state text

    cv2.putText(frame, f"Level: {level_up}", (500, 40), #Para ito sa leveling_up
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if state == DEAD:
        cv2.putText(frame, "YOU DIED", (200, 200), #position (x=200, y=200)
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4) #clean readable font, 2 → font scale (size), color → chosen based on state, 4 → text thickness
        cv2.putText(frame, "Press 'r' to restart or 'e' to exit", (175, 250), #position (x=200, y=200)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #clean readable font, 2 → font scale (size), color → chosen based on state, 4 → text thickness      


    cv2.imshow("Red Light Green Light (RLGL) - OpenCV", frame)

#-----------------------------------Quit or Restart----------------------------------------#
  
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and state == DEAD: #pag tinanggal natin yung state ==Dead, anytime pwede na tayo mag restart
        level_up = 0
        state = GREEN
        state_start_time = time.time()
        idle_start_time = None
        randomize_durations()

    elif key == ord('s'): #Game Start
        level_up = 0
        state = GREEN
        state_start_time = time.time()
        idle_start_time = None
        randomize_durations()

    elif key == ord('e'):
        print("End of Session! Thank you!")
        break
#-----------------------------------Closing----------------------------------------#

cap.release()
cv2.destroyAllWindows()
