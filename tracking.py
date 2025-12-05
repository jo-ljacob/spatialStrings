import cv2
import mediapipe as mp
import numpy as np
import globals

class TrackingPoint:
    def __init__(self, x, y, z):
        self.rawX = x
        self.rawY = y
        self.rawZ = z
    
        # Normalize all axes to units of meters
        self.normalizedX = self.rawX * max(globals.image.shape[0], 
                                 globals.image.shape[1]) / globals.image.shape[1]
        self.normalizedY = self.rawY * max(globals.image.shape[0],
                                 globals.image.shape[1]) / globals.image.shape[0]
        self.normalizedZ = None
        
        # Coordinates relative to head
        self.x = None
        self.y = None
        self.z = None
        
        # Convert to pixel coordinates
        self.pixelX = int(self.rawX * globals.image.shape[1])
        self.pixelY = int(self.rawY * globals.image.shape[0])
        self.size = None
        
        # Spherical coordinates
        self.rho = None
        self.theta = None
        self.phi = None
    
    # Normalization of face and hands must be different (empirically fitted)
    def normalizeFace(self):
        self.normalizedZ = 0.8 / (self.rawZ + 0.2)
        
    def normalizeHand(self):
        self.normalizedZ = -0.02 / (self.rawZ + 0.01)
        if self.normalizedZ < 0.15:
            self.normalizedZ = 0.15
        if self.normalizedZ > 2.5:
            self.normalizedZ = 2.5
        self.size = globals.SPRITE_SIZE/(self.normalizedZ+1e-10)
        
    # Transform coordinates of object to be in terms of a set origin
    def transform(self, other):
        self.x = -(other.normalizedX - self.normalizedX)
        self.y = other.normalizedY - self.normalizedY
        self.z = other.normalizedZ - self.normalizedZ - 2

    def toSpherical(self, forward, right):
        point = np.array([self.x, self.y, self.z], dtype=float)
        forward = np.array([forward.x, forward.y, forward.z], dtype=float)
        right = np.array([right.x, right.y, right.z], dtype=float)

        forward /= np.linalg.norm(forward)

        right -= np.dot(right, forward) * forward
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)

        rotationMatrix = np.array([right, up, forward])
        x, y, z = rotationMatrix @ point

        self.rho = np.linalg.norm([x, y, z])
        self.theta = np.arctan2(x, -z) % (2 * np.pi)
        self.phi = -np.arctan2(y, np.sqrt(x**2 + z**2))
            
        # Clamp phi to -45 degrees if it goes below that threshold
        if self.phi < -np.pi / 5:
            self.phi = -np.pi / 5

def distance(p1, p2):
    return ((p2.x-p1.x)**2+(p2.y-p1.y)**2)**0.5

def areFingersTouching(fingers, scale):
    landmarks = list(fingers.values())
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            if distance(landmarks[i], landmarks[j]) > scale * 0.5:
                return False
    return True

def initializeTracking():
    globals.face = mp.solutions.face_mesh.FaceMesh()
    globals.hands = mp.solutions.hands.Hands()
    
    head = TrackingPoint(0, 0, 0)
    head.normalizeFace()
    noseTip = TrackingPoint(0, 0, 0)
    noseTip.normalizeFace()
    rightEar = TrackingPoint(0, 0, 0)
    rightEar.normalizeFace()
    fingers = {}
    for landmark in [8, 4, 12, 16, 20]:
        fingers[landmark] = TrackingPoint(0,0,0)
        fingers[landmark].normalizeHand()
    scale = 1

    globals.trackingPositions = [head, noseTip, rightEar, fingers, scale]

def updateTracking():
    globals.image = globals.cap.read()[1]
    globals.image = cv2.cvtColor(cv2.flip(globals.image, 1), cv2.COLOR_BGR2RGB)

    faceResults = globals.face.process(globals.image)
    handResults = globals.hands.process(globals.image)
    
    globals.image = cv2.cvtColor(globals.image, cv2.COLOR_RGB2BGR)
    
    prevHead, prevNoseTip, prevRightEar, prevFingers, prevScale = globals.trackingPositions
    
    # Create face tracking points
    if faceResults.multi_face_landmarks:
        rawLeftEar = faceResults.multi_face_landmarks[0].landmark[234]
        rawRightEar = faceResults.multi_face_landmarks[0].landmark[454]
        
        # Simplify the head to a single point to serve as the origin
        head = TrackingPoint((rawLeftEar.x + rawRightEar.x) / 2, 
                             (rawLeftEar.y + rawRightEar.y) / 2,
                             (rawLeftEar.z + rawRightEar.z) / 2)
        head.normalizeFace()
        
        # Serves as forward vector
        rawNoseTip = faceResults.multi_face_landmarks[0].landmark[1]
        noseTip = TrackingPoint(rawNoseTip.x, rawNoseTip.y, rawNoseTip.z)
        noseTip.normalizeFace()
        noseTip.transform(head)

        # Serves as right vector
        rightEar = TrackingPoint(rawRightEar.x, rawRightEar.y, rawRightEar.z)
        rightEar.normalizeFace()
        rightEar.transform(head)
    else:
        head, noseTip, rightEar = prevHead, prevNoseTip, prevRightEar
    
    globals.isHandVisible = handResults.multi_hand_landmarks
    
    # Create finger tracking points
    fingers = {}
    for landmark in [8, 4, 12, 16, 20]:
        if globals.isHandVisible and handResults.multi_hand_landmarks[0]:
            raw = handResults.multi_hand_landmarks[0].landmark[landmark]
            fingers[landmark] = TrackingPoint(raw.x, raw.y, raw.z)
            fingers[landmark].normalizeHand()
            fingers[landmark].transform(head)
        else:
            fingers[landmark] = prevFingers[landmark]

    if globals.isHandVisible and handResults.multi_hand_landmarks[0]:
        fingers[8].toSpherical(noseTip, rightEar)
    else:
        fingers[8].rho = prevFingers[8].rho
        fingers[8].theta = prevFingers[8].theta
        fingers[8].phi = prevFingers[8].phi
    
    # Create a "scale" to determine threshold if fingers are touching
    if globals.isHandVisible:
        rawWrist = handResults.multi_hand_landmarks[0].landmark[0]
        wrist = TrackingPoint(rawWrist.x, rawWrist.y, rawWrist.z)
        wrist.normalizeHand()
        wrist.transform(head)
    
        rawKnuckle = handResults.multi_hand_landmarks[0].landmark[9]
        knuckle = TrackingPoint(rawKnuckle.x, rawKnuckle.y, rawKnuckle.z)
        knuckle.normalizeHand()
        knuckle.transform(head)
        
        scale = distance(wrist, knuckle)
    else:
        scale = prevScale

    if globals.isHandVisible:
        globals.areFingersTouching = areFingersTouching(fingers, scale)
    else:
        globals.areFingersTouching = False
        
    globals.trackingPositions = [head, noseTip, rightEar, fingers, scale]