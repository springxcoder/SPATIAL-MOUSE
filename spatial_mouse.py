import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import urllib.request



# THE VERNAL SPATIAL MOUSE

# NORMAL MODE:
#   Index finger              -> Cursor
#   Index + Middle together   -> Click / Drag

# SCROLL MODE:
#   Thumb + Middle together   -> Lock scrolling
#   Index moves UP            -> Scroll UP
#   Index moves DOWN          -> Scroll DOWN
#   Release Thumb + Middle    -> Return to cursor mode



# 1. MEDIA PIPE MODEL


MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MODEL_PATH = "hand_landmarker.task"


def download_model():

    if os.path.exists(MODEL_PATH):

        print("Hand model already exists.")

        return


    print("Downloading hand-tracking model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Hand model downloaded successfully!")


download_model()


# 2. SCREEN SETTINGS

CAMERA_TITLE = "Vernal Holograph - Spatial Mouse"

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

SAFE_X_MAX = SCREEN_WIDTH - 2
SAFE_Y_MAX = SCREEN_HEIGHT - 2


# 3. GESTURE THRESHOLDS

BASE_CLICK_START = 0.045
BASE_CLICK_RELEASE = 0.065

BASE_SCROLL_START = 0.045
BASE_SCROLL_RELEASE = 0.065


# 4. CURSOR FILTER SETTINGS

MIN_CUTOFF = 0.05
BETA = 0.70
D_CUTOFF = 1.0


# 5. SCROLL SETTINGS

SCROLL_MULTIPLIER = 50.0

SCROLL_DEAD_ZONE = 2.0


# 6. MEDIAPIPE SETUP

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.70,

    min_hand_presence_confidence=0.70,

    min_tracking_confidence=0.80
)


# 7. CAMERA

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Webcam could not open.")

    raise SystemExit


# 8. STATE VARIABLES

# Click state
clicking = False

mouse_is_down = False


# Scroll state
scroll_locked = False

last_scroll_y = None


# Timing
start_time = time.time()

last_frame_time = time.time()


# 9. ONE EURO FILTER STATE

x_filt = {

    "x_prev": None,

    "dx_prev": 0.0

}


y_filt = {

    "x_prev": None,

    "dx_prev": 0.0

}


# 10. HELPER FUNCTIONS

def distance_3d(a, b):

    return math.sqrt(

        (a.x - b.x) ** 2
        +
        (a.y - b.y) ** 2
        +
        (a.z - b.z) ** 2

    )


def low_pass_filter(alpha, x, previous):

    return (

        alpha * x
        +
        (1.0 - alpha) * previous

    )


def one_euro_filter(x, state, dt):

    # First frame
    if state["x_prev"] is None:

        state["x_prev"] = x

        return x


    # Velocity

    if dt > 0:

        dx = (
            x - state["x_prev"]
        ) / dt

    else:

        dx = 0.0


    # Filter velocity

    if dt > 0:

        alpha_d = 1.0 / (

            1.0
            +
            1.0 / (
                2.0
                * math.pi
                * D_CUTOFF
                * dt
            )

        )

    else:

        alpha_d = 1.0


    dx_hat = low_pass_filter(

        alpha_d,

        dx,

        state["dx_prev"]

    )


    state["dx_prev"] = dx_hat


    # Dynamic cutoff

    cutoff = (

        MIN_CUTOFF
        +
        BETA * abs(dx_hat)

    )


    # Filter position

    if dt > 0:

        alpha = 1.0 / (

            1.0
            +
            1.0 / (
                2.0
                * math.pi
                * cutoff
                * dt
            )

        )

    else:

        alpha = 1.0


    x_hat = low_pass_filter(

        alpha,

        x,

        state["x_prev"]

    )


    state["x_prev"] = x_hat


    return x_hat


# 11. MAIN LOOP

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        # READ CAMERA

        success, frame = cap.read()


        if not success:

            continue


        # Mirror camera
        frame = cv2.flip(frame, 1)


        height, width, _ = frame.shape


        # 
        # MEDIAPIPE IMAGE
        # 

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame

        )


        # TIME

        current_time = time.time()


        dt = (

            current_time
            -
            last_frame_time

        )


        last_frame_time = current_time


        timestamp_ms = int(

            (current_time - start_time)
            * 1000

        )


        # DETECT HAND

        result = landmarker.detect_for_video(

            mp_image,

            timestamp_ms

        )


        # DEFAULT STATUS

        status = "READY"

        status_color = (

            0,
            255,
            0

        )


        # HAND FOUND

        if (

            result.hand_landmarks
            and
            len(result.hand_landmarks) > 0

        ):

            landmarks = result.hand_landmarks[0]


        
            # GET IMPORTANT LANDMARKS
        

            thumb = landmarks[4]

            index_base = landmarks[5]

            index = landmarks[8]

            middle = landmarks[12]


        
            # HAND SCALE
        

            hand_scale = distance_3d(

                index,

                index_base

            )


            scale_factor = max(

                0.5,

                min(

                    1.5,

                    hand_scale / 0.15

                )

            )


        
            # ADAPTIVE THRESHOLDS
        

            click_start = (

                BASE_CLICK_START
                * scale_factor

            )


            click_release = (

                BASE_CLICK_RELEASE
                * scale_factor

            )


            scroll_start = (

                BASE_SCROLL_START
                * scale_factor

            )


            scroll_release = (

                BASE_SCROLL_RELEASE
                * scale_factor

            )


        
            # GESTURE DISTANCES
        

            index_middle_dist = distance_3d(

                index,

                middle

            )


            thumb_middle_dist = distance_3d(

                thumb,

                middle

            )


        
            # MAP INDEX → SCREEN
        

            CAM_BOX_X_MIN = 0.25
            CAM_BOX_X_MAX = 0.75

            CAM_BOX_Y_MIN = 0.25
            CAM_BOX_Y_MAX = 0.75


            norm_x = (

                index.x
                -
                CAM_BOX_X_MIN

            ) / (

                CAM_BOX_X_MAX
                -
                CAM_BOX_X_MIN

            )


            norm_y = (

                index.y
                -
                CAM_BOX_Y_MIN

            ) / (

                CAM_BOX_Y_MAX
                -
                CAM_BOX_Y_MIN

            )


            # Clamp
            norm_x = max(

                0.0,

                min(1.0, norm_x)

            )


            norm_y = max(

                0.0,

                min(1.0, norm_y)

            )


        
            # SCREEN POSITION
        

            target_x = (

                norm_x
                * SCREEN_WIDTH

            )


            target_y = (

                norm_y
                * SCREEN_HEIGHT

            )


        
            # FILTER INDEX POSITION
        

            filtered_x = one_euro_filter(

                target_x,

                x_filt,

                dt

            )


            filtered_y = one_euro_filter(

                target_y,

                y_filt,

                dt

            )


            # Safe screen limits

            filtered_x = max(

                1,

                min(

                    SAFE_X_MAX,

                    filtered_x

                )

            )


            filtered_y = max(

                1,

                min(

                    SAFE_Y_MAX,

                    filtered_y

                )

            )


        
            # SCROLL MODE
        

            if scroll_locked:

        
                # THUMB + MIDDLE RELEASED?
        

                if (

                    thumb_middle_dist
                    >= scroll_release

                ):

                    # Exit scroll mode

                    scroll_locked = False

                    last_scroll_y = None


                    status = "SCROLL RELEASED"

                    status_color = (

                        0,
                        255,
                        0

                    )


                else:

                
                    # SCROLL MODE ACTIVE
                

                    status = "SCROLL MODE"

                    status_color = (

                        255,
                        0,
                        255

                    )


            
                    # Compare current index Y to previous index Y
            

                    if last_scroll_y is not None:

                        delta_y = (

                            filtered_y
                            -
                            last_scroll_y

                        )


                        # Ignore tiny movements

                        if abs(delta_y) >= SCROLL_DEAD_ZONE:
                            # INDEX MOVED U

                            if delta_y < 0:

                                ticks = int(

                                    abs(delta_y)
                                    * SCROLL_MULTIPLIER

                                )


                                if ticks > 0:

                                    pyautogui.scroll(

                                        ticks

                                    )


                                status = "SCROLLING UP"

                            # INDEX MOVED DOW

                            elif delta_y > 0:

                                ticks = int(

                                    abs(delta_y)
                                    * SCROLL_MULTIPLIER

                                )


                                if ticks > 0:

                                    pyautogui.scroll(

                                        -ticks

                                    )


                                status = "SCROLLING DOWN"


                    # Save current index position

                    last_scroll_y = filtered_y


        
            # NORMAL MODE
        

            else:

            
                # CHECK SCROLL GESTURE FIRST
            

                if (

                    thumb_middle_dist
                    <= scroll_start

                ):

                    # Enter scroll mode

                    scroll_locked = True


                    # Save the exact index position
                    # at the moment scrolling begins

                    last_scroll_y = filtered_y


                    status = "SCROLL LOCKED"

                    status_color = (

                        255,
                        120,
                        0

                    )


            
                # CLICK / DRAG
            

                elif clicking:


                    # RELEASE CLICK


                    if (

                        index_middle_dist
                        >= click_release

                    ):

                        clicking = False


                        if mouse_is_down:

                            pyautogui.mouseUp()

                            mouse_is_down = False


                        status = "RELEASE"



                    # CONTINUE DRAGGING


                    else:

                        status = "DRAGGING"

                        status_color = (

                            0,
                            255,
                            255

                        )


                        if not mouse_is_down:

                            pyautogui.mouseDown()

                            mouse_is_down = True


            
                # START CLICK
            

                elif (

                    index_middle_dist
                    <= click_start

                ):

                    clicking = True


                    if not mouse_is_down:

                        mouse_is_down = True

                        pyautogui.mouseDown()


                    status = "CLICK"

                    status_color = (

                        0,
                        255,
                        255

                    )


            
                # NORMAL CURSOR MOVEMENT
            

                else:

                    status = "MOVING"


                    pyautogui.moveTo(

                        int(filtered_x),

                        int(filtered_y),

                        duration=0

                    )


        
            # DRAW INDEX POSITION
        

            index_px_x = int(

                index.x
                * width

            )


            index_px_y = int(

                index.y
                * height

            )


            cv2.circle(

                frame,

                (
                    index_px_x,
                    index_px_y
                ),

                10,

                status_color,

                -1

            )


        
            # DRAW CONTROL BOX
        

            box_x1 = int(

                CAM_BOX_X_MIN
                * width

            )


            box_x2 = int(

                CAM_BOX_X_MAX
                * width

            )


            box_y1 = int(

                CAM_BOX_Y_MIN
                * height

            )


            box_y2 = int(

                CAM_BOX_Y_MAX
                * height

            )


            cv2.rectangle(

                frame,

                (
                    box_x1,
                    box_y1
                ),

                (
                    box_x2,
                    box_y2
                ),

                (
                    255,
                    255,
                    255
                ),

                2

            )


        # 
        # NO HAND
        # 

        else:

            status = "NO HAND"

            status_color = (

                0,
                0,
                255

            )


    
            # Safety: release mouse
    

            if mouse_is_down:

                pyautogui.mouseUp()

                mouse_is_down = False


            clicking = False


    
            # Reset scroll
    

            scroll_locked = False

            last_scroll_y = None


        # HUD

        cv2.putText(

            frame,

            f"STATUS: {status}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            status_color,

            2

        )


        cv2.putText(

            frame,

            "INDEX = CURSOR",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            "INDEX + MIDDLE = CLICK / DRAG",

            (20, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            "THUMB + MIDDLE = SCROLL LOCK",

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            "INDEX UP/DOWN = SCROLL",

            (20, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            "PRESS Q TO EXIT",

            (20, height - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1

        )


        # SHOW CAMERA

        cv2.imshow(

            CAMERA_TITLE,

            frame

        )


        # EXIT

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# CLEANUP

cap.release()

cv2.destroyAllWindows()

print("Floating Mousestopped.")
