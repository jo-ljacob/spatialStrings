import cv2
import numpy as np
import pyfar as pf

SPRITE_SIZE = 0.2
SOUNDS_DIR = "./sounds"
IMAGES_DIR = "./images"
HRTF_UPDATE_THRESHOLD_RADIAN = 0.1
MIN_DISTANCE = 0.25
VOL_CAP = 0.2
WINDOW_WIDTH = 805
WINDOW_HEIGHT = 670

cap = cv2.VideoCapture(0)
face = None
hand = None
trackingPositions = []
image = np.zeros((480, 640, 3), dtype=np.uint8)
isPlacingOn = True
isHandVisible = False
areFingersTouching = False
wereFingersTouching = False
hrtf, sofa = pf.io.read_sofa('vikingHRTF.sofa', verify=False)[:2]
instruments = []
instrumentsPlaying = []
undoStack = []
redoStack = []
currentInstrument = None
playbackPosition = 0
stream = None
volume = 0.5