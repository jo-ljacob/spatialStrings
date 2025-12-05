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
        if wavSampleRate != globals.hrtf.sampling_rate:
            numSamples = int(len(wav) * globals.hrtf.sampling_rate / wavSampleRate)
            wav = resample(wav, numSamples)
        self.cachedAudio = wav

    def getHRTF(self):
        hrtfCoords = globals.sofa.get_sph()
        
        # Find the closest point to the ideal coordinates in the HRTF file
        distances = ((hrtfCoords[:, 0] - self.theta)**2 + 
                    (hrtfCoords[:, 1] - self.phi)**2)**0.5
        
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

class RealtimeConvolver:
    # Taken from AI to achieve FFT convolution
    """
    Partitioned overlap-add convolver for real-time HRTF processing.
    Based on uniformly partitioned convolution algorithm.
    """
    def __init__(self, ir_left, ir_right, block_size=512, partition_size=128):
        self.block_size = block_size
        self.partition_size = partition_size
        
        # CRITICAL FIX #1: Pad IRs to multiple of partition size
        ir_len = len(ir_left)
        num_partitions = int(np.ceil(ir_len / partition_size))
        padded_len = num_partitions * partition_size
        
        self.ir_left = np.pad(ir_left, (0, padded_len - ir_len))
        self.ir_right = np.pad(ir_right, (0, padded_len - ir_len))
        
        # Pre-compute FFT of IR partitions
        self.num_partitions = num_partitions
        fft_size = partition_size * 2  # Must be 2x for linear convolution
        
        self.ir_left_fft = []
        self.ir_right_fft = []
        
        # CRITICAL FIX #2: Zero-pad each partition before FFT
        for i in range(num_partitions):
            start = i * partition_size
            end = start + partition_size
            # Zero-pad to fft_size for proper linear convolution
            left_partition = np.pad(self.ir_left[start:end], (0, partition_size))
            right_partition = np.pad(self.ir_right[start:end], (0, partition_size))
            self.ir_left_fft.append(np.fft.rfft(left_partition, fft_size))
            self.ir_right_fft.append(np.fft.rfft(right_partition, fft_size))
        
        # Frequency-domain delay lines (FDL)
        self.fdl_left = [np.zeros(fft_size // 2 + 1, dtype=complex) for _ in range(num_partitions)]
        self.fdl_right = [np.zeros(fft_size // 2 + 1, dtype=complex) for _ in range(num_partitions)]
        
        # Overlap-add buffers
        self.overlap_left = np.zeros(partition_size)
        self.overlap_right = np.zeros(partition_size)
        
        # Crossfade state
        self.crossfade_active = False
        self.crossfade_samples = 0
        self.crossfade_duration = 512  # samples
        self.old_convolver = None
        
    def process(self, input_block):
        """Process one block using partitioned convolution"""
        # CRITICAL FIX #3: Handle block_size != partition_size properly
        output_left = np.zeros(len(input_block))
        output_right = np.zeros(len(input_block))
        
        # Process in chunks of partition_size
        num_chunks = int(np.ceil(len(input_block) / self.partition_size))
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.partition_size
            end = min(start + self.partition_size, len(input_block))
            chunk = input_block[start:end]
            
            # CRITICAL FIX #4: Always use full partition_size (zero-pad if needed)
            if len(chunk) < self.partition_size:
                chunk = np.pad(chunk, (0, self.partition_size - len(chunk)))
            
            fft_size = self.partition_size * 2
            
            # Zero-pad input chunk before FFT
            chunk_padded = np.pad(chunk, (0, self.partition_size))
            chunk_fft = np.fft.rfft(chunk_padded, fft_size)
            
            # CRITICAL FIX #5: Shift FDL correctly (newest at index 0)
            self.fdl_left.insert(0, chunk_fft)
            self.fdl_left.pop()
            self.fdl_right.insert(0, chunk_fft)
            self.fdl_right.pop()
            
            # Multiply and accumulate
            acc_left = np.zeros(fft_size // 2 + 1, dtype=complex)
            acc_right = np.zeros(fft_size // 2 + 1, dtype=complex)
            
            for i in range(self.num_partitions):
                acc_left += self.fdl_left[i] * self.ir_left_fft[i]
                acc_right += self.fdl_right[i] * self.ir_right_fft[i]
            
            # IFFT
            out_left = np.fft.irfft(acc_left, fft_size)
            out_right = np.fft.irfft(acc_right, fft_size)
            
            # CRITICAL FIX #6: Overlap-add - first partition_size samples + overlap
            out_chunk_left = out_left[:self.partition_size] + self.overlap_left
            out_chunk_right = out_right[:self.partition_size] + self.overlap_right
            
            # Store second half as overlap for next iteration
            self.overlap_left = out_left[self.partition_size:fft_size]
            self.overlap_right = out_right[self.partition_size:fft_size]
            
            # Copy to output (handle partial last chunk)
            out_len = min(self.partition_size, len(input_block) - start)
            output_left[start:start + out_len] = out_chunk_left[:out_len]
            output_right[start:start + out_len] = out_chunk_right[:out_len]
        
        return output_left, output_right
    
    def update_ir(self, new_ir_left, new_ir_right):
        """Update HRTF with crossfade"""
        # Save old convolver state for crossfading
        self.old_convolver = RealtimeConvolver(
            self.ir_left[:len(new_ir_left)], 
            self.ir_right[:len(new_ir_right)],
            self.block_size,
            self.partition_size
        )
        # CRITICAL FIX #7: Deep copy state to old convolver
        self.old_convolver.fdl_left = [fdl.copy() for fdl in self.fdl_left]
        self.old_convolver.fdl_right = [fdl.copy() for fdl in self.fdl_right]
        self.old_convolver.overlap_left = self.overlap_left.copy()
        self.old_convolver.overlap_right = self.overlap_right.copy()
        
        # Reinitialize with new IR
        old_block_size = self.block_size
        old_partition_size = self.partition_size
        self.__init__(new_ir_left, new_ir_right, old_block_size, old_partition_size)
        
        self.crossfade_active = True
        self.crossfade_samples = 0

def commit():
    globals.undoStack.append([inst for inst in globals.instrumentsPlaying])
    globals.redoStack.clear()

def spawnRandomInstruments():
    commit()
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
            commit()
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
        
        # Distance attenuation
        safeDistance = max(instrument.rho, globals.MIN_DISTANCE)
        gain = globals.volume * globals.VOL_CAP / safeDistance
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