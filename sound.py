import numpy as np
import sounddevice as sd
import soundfile as sf
import os
from scipy.signal import resample
import globals
import random

class Instrument():
    def __init__(self, wav):
        self.normalizedX = None
        self.normalizedY = None
        self.normalizedZ = None
        
        self.x = None
        self.y = None
        self.z = None

        self.pixelX = None
        self.pixelY = None
        self.size = None
        
        self.rho = None
        self.theta = None
        self.phi = None
        
        self.wav = wav
        self.cachedAudio = None
        
        self.convolver = None
        self.lastTheta = None
        self.lastPhi = None
        
    def inherit(self, other):
        self.normalizedX = other.normalizedX
        self.normalizedY = other.normalizedY
        self.normalizedZ = other.normalizedZ
        
        self.x = other.x
        self.y = other.y
        self.z = other.z
        
        self.pixelX = other.pixelX
        self.pixelY = other.pixelY
        self.size = other.size 
        
        self.rho = other.rho
        self.theta = other.theta
        self.phi = other.phi
        
    def transform(self, other):
        self.x = -(other.normalizedX - self.normalizedX)
        self.y = other.normalizedY - self.normalizedY
        self.z = other.normalizedZ - self.normalizedZ - 2
        
    def cacheAudio(self):
        wav, wavSampleRate = sf.read(self.wav)
        wav = np.mean(wav, axis=1) # Make audio mono
        if wavSampleRate != globals.hrtf.sampling_rate:
            numSamples = int(len(wav) * globals.hrtf.sampling_rate / wavSampleRate)
            wav = resample(wav, numSamples)
        self.cachedAudio = wav

    def getHRTF(self):
        hrtf_coords_sph = globals.sofa.get_sph()
        
        # Find the closest point to the ideal coordinates in the HRTF file
        distances = ((hrtf_coords_sph[:, 0] - self.theta)**2 + 
                    (hrtf_coords_sph[:, 1] - self.phi)**2)**0.5
        
        nearest = np.argmin(distances)
        
        hrir = globals.hrtf[nearest]
        left = hrir.time[0, :]
        right = hrir.time[1, :]
        
        return left, right

    def initializeConvolver(self, block_size=2048):
        left, right = self.getHRTF()
        self.convolver = RealtimeConvolver(left, right, block_size, partition_size=128)
        self.lastTheta = self.theta
        self.lastPhi = self.phi
        
    def updateHRTF(self):
        left, right = self.getHRTF()
        self.convolver.update_ir(left, right)
        self.lastTheta = self.theta
        self.lastPhi = self.phi
        
    def toSpherical(self, forward, right):
        point = np.array([self.x, self.y, self.z], dtype=float)
        forward = np.array([forward.x, forward.y, forward.z], dtype=float)
        right = np.array([right.x, right.y, right.z], dtype=float)
        
        forward = np.array(forward, dtype=float)
        forward /= np.linalg.norm(forward)
        
        right = np.array(right, dtype=float)
        right -= np.dot(right, forward) * forward  # Make orthogonal
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)
        
        rotationMatrix = np.array([right, up, forward])
        pointRotated = rotationMatrix @ point
        x, y, z = pointRotated

        self.rho = np.linalg.norm(pointRotated)
        self.theta = np.arctan2(-x, -z)
        self.phi = -np.arcsin(y / self.rho)

class RealtimeConvolver:
    # Taken from AI to achieve FFT convolution
    def __init__(self, ir_left, ir_right, block_size=512, partition_size=128):
        self.block_size = block_size
        self.partition_size = partition_size
        
        # Pad IRs to multiple of partition size
        ir_len = len(ir_left)
        num_partitions = int(np.ceil(ir_len / partition_size))
        padded_len = num_partitions * partition_size
        
        self.ir_left = np.pad(ir_left, (0, padded_len - ir_len))
        self.ir_right = np.pad(ir_right, (0, padded_len - ir_len))
        
        # Pre-compute FFT of IR partitions
        self.num_partitions = num_partitions
        fft_size = partition_size * 2
        
        self.ir_left_fft = []
        self.ir_right_fft = []
        
        for i in range(num_partitions):
            start = i * partition_size
            end = start + partition_size
            self.ir_left_fft.append(np.fft.rfft(self.ir_left[start:end], fft_size))
            self.ir_right_fft.append(np.fft.rfft(self.ir_right[start:end], fft_size))
        
        # Overlap-add buffers
        self.fdl_left = [np.zeros(fft_size // 2 + 1, dtype=complex) for _ in range(num_partitions)]
        self.fdl_right = [np.zeros(fft_size // 2 + 1, dtype=complex) for _ in range(num_partitions)]
        self.overlap_left = np.zeros(partition_size)
        self.overlap_right = np.zeros(partition_size)
        
        # Crossfade state
        self.crossfade_active = False
        self.crossfade_samples = 0
        self.crossfade_duration = 512  # samples
        self.old_convolver = None
        
    def process(self, input_block):
        """Process one block using partitioned convolution"""
        # Process in chunks of partition_size
        num_chunks = len(input_block) // self.partition_size
        output_left = np.zeros(len(input_block))
        output_right = np.zeros(len(input_block))
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.partition_size
            end = start + self.partition_size
            chunk = input_block[start:end]
            
            # FFT of input
            fft_size = self.partition_size * 2
            chunk_fft = np.fft.rfft(chunk, fft_size)
            
            # Shift FDL (frequency domain delay line)
            self.fdl_left = [chunk_fft] + self.fdl_left[:-1]
            self.fdl_right = [chunk_fft] + self.fdl_right[:-1]
            
            # Multiply and accumulate
            acc_left = np.zeros(fft_size // 2 + 1, dtype=complex)
            acc_right = np.zeros(fft_size // 2 + 1, dtype=complex)
            
            for i in range(self.num_partitions):
                acc_left += self.fdl_left[i] * self.ir_left_fft[i]
                acc_right += self.fdl_right[i] * self.ir_right_fft[i]
            
            # IFFT
            out_left = np.fft.irfft(acc_left)
            out_right = np.fft.irfft(acc_right)
            
            # Overlap-add
            output_left[start:end] = out_left[:self.partition_size] + self.overlap_left
            output_right[start:end] = out_right[:self.partition_size] + self.overlap_right
            
            self.overlap_left = out_left[self.partition_size:]
            self.overlap_right = out_right[self.partition_size:]
        
        return output_left, output_right
    
    def update_ir(self, new_ir_left, new_ir_right):
        """Update HRTF with crossfade"""
        self.old_convolver = RealtimeConvolver(
            self.ir_left[:len(new_ir_left)], 
            self.ir_right[:len(new_ir_right)],
            self.block_size,
            self.partition_size
        )
        self.old_convolver.fdl_left = self.fdl_left.copy()
        self.old_convolver.fdl_right = self.fdl_right.copy()
        self.old_convolver.overlap_left = self.overlap_left.copy()
        self.old_convolver.overlap_right = self.overlap_right.copy()
        
        self.__init__(new_ir_left, new_ir_right, self.block_size, self.partition_size)
        self.crossfade_active = True
        self.crossfade_samples = 0
        
def spawnRandomInstruments():
    globals.instrumentsPlaying = []

    head = globals.trackingPositions[0]
    
    for _ in range(5):
        randomWav = random.choice(globals.instruments)
        newInstrument = Instrument(randomWav)
   
        newInstrument.rho = random.uniform(0.25, 0.6)
        newInstrument.theta = random.uniform(-np.pi, np.pi)
        newInstrument.phi = random.uniform(-np.pi/2, np.pi/2) 
        
        newInstrument.x = -newInstrument.rho * np.cos(newInstrument.phi) * np.sin(newInstrument.theta)
        newInstrument.y = -newInstrument.rho * np.sin(newInstrument.phi)
        newInstrument.z = -newInstrument.rho * np.cos(newInstrument.phi) * np.cos(newInstrument.theta)

        newInstrument.normalizedX = head.normalizedX + newInstrument.x
        newInstrument.normalizedY = head.normalizedY - newInstrument.y
        newInstrument.normalizedZ = head.normalizedZ - newInstrument.z - 2

        newInstrument.rawX = newInstrument.normalizedX / (max(globals.image.shape[0],
            globals.image.shape[1]) / globals.image.shape[1])
        newInstrument.rawY = newInstrument.normalizedY / (max(globals.image.shape[0],
            globals.image.shape[1]) / globals.image.shape[0])
        
        newInstrument.pixelX = int(newInstrument.rawX * globals.image.shape[1])
        newInstrument.pixelY = int(newInstrument.rawY * globals.image.shape[0])
        newInstrument.size = (globals.SPRITE_SIZE)/(newInstrument.rho+1e-10)

        newInstrument.cacheAudio()
        newInstrument.initializeConvolver()
        
        globals.instrumentsPlaying.append(newInstrument)
        
def initializeAudio():
    # Compile a list of instrument choices
    for sound in os.listdir(globals.SOUNDS_DIR):
        if sound.lower().endswith(".wav"):
            globals.instruments.append(os.path.join(globals.SOUNDS_DIR, sound))
    globals.instruments.sort()

    globals.currentInstrument = globals.instruments[0]
        
    globals.stream = sd.OutputStream(
        samplerate=globals.hrtf.sampling_rate,
        channels=2,
        callback=audioCallback,
        blocksize=512)
    globals.stream.start()

def updateSources():
    head, noseTip, rightEar, fingers, scale = globals.trackingPositions
    
    # Place an instrument when fingers touch
    if globals.isPlacingOn and globals.isHandVisible:
        if globals.areFingersTouching and not globals.wereFingersTouching:
            newInstrument = Instrument(globals.currentInstrument)
            newInstrument.inherit(fingers[8])
            newInstrument.cacheAudio()
            newInstrument.initializeConvolver()
            globals.instrumentsPlaying.append(newInstrument)
            globals.wereFingersTouching = True
        
        elif not globals.areFingersTouching and globals.wereFingersTouching:
            globals.wereFingersTouching = False
    
    for instrument in globals.instrumentsPlaying:
        instrument.transform(head)
        instrument.toSpherical(noseTip, rightEar)
        
        # Update only when position significantly changed
        if (abs(instrument.theta - instrument.lastTheta) > globals.HRTF_UPDATE_THRESHOLD_RADIAN or 
            abs(instrument.phi - instrument.lastPhi) > globals.HRTF_UPDATE_THRESHOLD_RADIAN):
            instrument.updateHRTF()

def audioCallback(outdata, frames, time, status):
    output = np.zeros((frames, 2))
    
    if not globals.instrumentsPlaying:
        outdata[:] = output
        return
    
    audioLength = len(globals.instrumentsPlaying[0].cachedAudio)
    startPos = globals.playbackPosition % audioLength
    endPos = startPos + frames
    
    for instrument in globals.instrumentsPlaying:     
        if endPos <= audioLength:
            audioChunk = instrument.cachedAudio[startPos:endPos]
        else:
            audioChunk = np.concatenate([
                instrument.cachedAudio[startPos:], 
                instrument.cachedAudio[:endPos - audioLength]])
        
        # Gain usually proportional to just 1/distance, but wanted to emphasize distance attenuation
        safeDistance = max(instrument.rho, globals.MIN_DISTANCE)
        gain = globals.volume * globals.VOL_CAP / safeDistance**2
        audioChunk = audioChunk * gain
        
        left, right = instrument.convolver.process(audioChunk)

        if instrument.convolver.crossfade_active and instrument.convolver.old_convolver is not None:
            oldLeft, oldRight = instrument.convolver.old_convolver.process(audioChunk)
            
            fadeEnd = min(instrument.convolver.crossfade_samples + frames, 
                        instrument.convolver.crossfade_duration)
            fadeLen = fadeEnd - instrument.convolver.crossfade_samples
            
            if fadeLen > 0:
                fadeCurve = np.linspace(0, 1, fadeLen)
                left[:fadeLen] = oldLeft[:fadeLen] * (1 - fadeCurve) + left[:fadeLen] * fadeCurve
                right[:fadeLen] = oldRight[:fadeLen] * (1 - fadeCurve) + right[:fadeLen] * fadeCurve
            
            instrument.convolver.crossfade_samples += frames
            if instrument.convolver.crossfade_samples >= instrument.convolver.crossfade_duration:
                instrument.convolver.crossfade_active = False
                instrument.convolver.old_convolver = None
        
        output[:, 0] += left
        output[:, 1] += right
        
    globals.playbackPosition += frames
    
    # Clip if necessary
    outdata[:] = np.tanh(output)