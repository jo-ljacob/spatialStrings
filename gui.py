import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import globals
import os
from sound import commit, spawnRandomInstruments

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title('spatialStrings')
        
        self.root.geometry(f'{globals.WINDOW_WIDTH}x{globals.WINDOW_HEIGHT}')
        self.root.resizable(False, False)
        
        self.root.config(bg='#eeeeff')
        
        # Top header
        headerFrame = tk.Frame(self.root, bg='#eeeeff', height=60, 
                               width=globals.WINDOW_WIDTH, relief='solid')
        headerFrame.place(x=0, y=0, width=globals.WINDOW_WIDTH, height=60)
        headerFrame.pack_propagate(False)
        
        # Title
        titleLabel = tk.Label(headerFrame, 
                              text="spatialStrings", 
                              font=('Times New Roman', 24, 'bold'), 
                              bg='#eeeeff', 
                              fg="black")
        titleLabel.pack(side='left', padx=10, pady=10)
        
        # Instructions
        tk.Label(headerFrame, 
                 text="Use headphones. Pinch fingers together to place an instrument.", 
                 bg='#eeeeff', 
                 font=('Times New Roman', 12, 'bold')).pack(side='right', padx=10, pady=(10,0))
        
        bar = tk.Frame(self.root, bg='black', height=2)
        bar.place(x=10, y=50, width=globals.WINDOW_WIDTH-20, height=2)
        
        # Controls
        controlY = 75
        controlFrame = tk.Frame(self.root, bg='#eeeeff', height=50, relief='flat')
        controlFrame.place(x=0, y=controlY, width=globals.WINDOW_WIDTH, height=50)
        controlFrame.pack_propagate(False)

        btnStyle = {'font': ('Arial', 9), 'relief': 'solid', 'borderwidth': 1, 
                    'padx': 10, 'pady': 5, 'bg': "#e0e0e0"}
        
        tk.Button(controlFrame, text="Clear", 
                  command=self.clearInstruments, 
                  **btnStyle).pack(side='left', padx=(12, 5), pady=10)
        tk.Button(controlFrame, text="Pop Last", 
                  command=self.popLast, 
                  **btnStyle).pack(side='left', padx=5, pady=10)
        
        tk.Button(controlFrame, text="Undo", 
                  command=self.undo, 
                  **btnStyle).pack(side='left', padx=(20,5), pady=10)
        tk.Button(controlFrame, text="Redo", 
                  command=self.redo, 
                  **btnStyle).pack(side='left', padx=5, pady=10)
        
        self.placementVar = tk.BooleanVar(value=True)
        tk.Checkbutton(controlFrame, text="Placement", variable=self.placementVar, 
                      command=self.togglePlacement,
                      font=('Times New Roman', 12), 
                      bg='#eeeeff').pack(side='left', padx=(15,10))
        
        tk.Label(controlFrame, text="Volume:", 
                 bg='#eeeeff', 
                 font=('Times New Roman', 12)).pack(side='left', padx=(10, 5))
        self.volumeSlider = tk.Scale(controlFrame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     command=self.onVolumeChange, length=190,
                                     bg='#eeeeff', highlightthickness=0, showvalue=False)
        self.volumeSlider.set(int(globals.volume * 100))
        self.volumeSlider.pack(side='left')
        
        tk.Button(controlFrame, text="Random", command=spawnRandomInstruments, **btnStyle).pack(side='right', padx=12, pady=10)
        
        # Content
        contentY = 130
        contentHeight = 500

        # Camera
        canvasWidth = 660
        canvasX = globals.WINDOW_WIDTH - canvasWidth - 10
        
        # Main canvas frame with border
        canvasFrame = tk.Frame(self.root, bg='#eeeac8', relief='solid', borderwidth=2)
        canvasFrame.place(x=canvasX, y=contentY, width=canvasWidth, height=contentHeight)

        self.videoWidth = 640
        self.videoHeight = 480
        
        self.videoLabel = tk.Label(canvasFrame, bg='#eeeac8', 
                                   relief='solid', highlightbackground='blue', 
                                   highlightthickness=1)
        self.videoLabel.place(x=8, y=8, width=self.videoWidth, height=self.videoHeight)

        # Instruments
        sidebarWidth = globals.WINDOW_WIDTH - canvasWidth - 20
        sidebarFrame = tk.Frame(self.root, bg='black', width=sidebarWidth, 
                                relief='solid', borderwidth=0)
        sidebarFrame.place(x=10, y=contentY, width=sidebarWidth, height=contentHeight)
        sidebarFrame.pack_propagate(False)
        
        # Instrument list
        self.instrumentsFrame = tk.Frame(sidebarFrame, bg='#eeeeff')
        self.instrumentsFrame.pack(side='left', fill='both', expand=True)
        
        self.createInstrumentButtons(self.instrumentsFrame)
        
        # Status
        statusY = contentY + contentHeight
        statusFrame = tk.Frame(self.root, bg='#eeeeff', height=40)
        statusFrame.place(x=0, y=statusY, width=globals.WINDOW_WIDTH, height=40)
        
        tk.Label(statusFrame, text="Status:", bg='#eeeeff', 
                font=('Times New Roman', 12, 'bold')).pack(side='left', padx=7)
        
        self.statusLabel = tk.Label(statusFrame, text="Ready", bg='#eeeeff',
                                     font=('Times New Roman', 12), fg='#00ff37')
        self.statusLabel.pack(side='left')
        
        # Instrument count
        self.countLabel = tk.Label(statusFrame, text="Instruments: 0", 
                                    bg='#eeeeff', font=('Times New Roman', 12))
        self.countLabel.pack(side='right', padx=7)
        
        self.photo = None
        
    def createInstrumentButtons(self, parent):
        instrumentList = ['Cello', 'Contrabass', 'Viola', 'Violin']
        
        self.instrumentImages = []
        self.instrumentButtons = []
        
        for i in range(len(instrumentList)):
            name = instrumentList[i]
            imagePath = os.path.join(globals.IMAGES_DIR, f"{name}.png")
            
            bgColor = '#e0e0e0'
            
            containerFrame = tk.Frame(parent, bg=bgColor, height=125)
            containerFrame.pack(fill='x')
            containerFrame.pack_propagate(False)
            
            leftBorder = tk.Frame(containerFrame, bg='black', width=2)
            leftBorder.place(x=0, y=0, width=2, height=125)
            
            topBorder = tk.Frame(containerFrame, bg='black', height=2)
            topBorder.pack(fill='x', side='top')
            
            if i == len(instrumentList) - 1:
                bottomBorder = tk.Frame(containerFrame, bg='black', height=2)
                bottomBorder.pack(fill='x', side='bottom')
            
            btnFrame = tk.Frame(containerFrame, bg=bgColor)
            btnFrame.pack(fill='both', expand=True, padx=(2, 0), pady=(2, 0 if i < len(instrumentList)-1 else 2))
            
            img = Image.open(imagePath)
            img = img.resize((80, 80), Image.Resampling.LANCZOS)
            photoImg = ImageTk.PhotoImage(img)
            self.instrumentImages.append(photoImg)
            
            imageLabel = tk.Label(btnFrame, image=photoImg, bg=bgColor)
            imageLabel.pack(pady=(10, 5))
            
            nameLabel = tk.Label(btnFrame, text=name.upper(), 
                                font=('Times New Roman', 10), 
                                bg=bgColor, fg='blue')
            nameLabel.pack(pady=(0, 5))
            
            # Store reference to widgets for updating
            self.instrumentButtons.append({
                'container': containerFrame,
                'btnFrame': btnFrame,
                'imageLabel': imageLabel,
                'nameLabel': nameLabel,
                'name': name
            })
            
            # Make clickable
            for widget in [btnFrame, imageLabel, nameLabel]:
                widget.bind('<Button-1>', lambda e, idx=i: self.selectInstrumentByIndex(idx))
        
        # Wait until globals are loaded
        self.root.after(10, self.updateInstrumentSelection)
    
    def selectInstrumentByIndex(self, idx):
        if idx < len(globals.instruments):
            globals.currentInstrument = globals.instruments[idx]
            self.updateInstrumentSelection()
    
    def updateInstrumentSelection(self):
        currentName = os.path.splitext(os.path.basename(globals.currentInstrument))[0]
        
        for btn in self.instrumentButtons:
            if btn['name'] == currentName: # Currently selected
                btn['container'].config(bg='white')
                btn['btnFrame'].config(bg='white')
                btn['imageLabel'].config(bg='white')
                btn['nameLabel'].config(bg='white')
            else:
                btn['container'].config(bg='#e0e0e0')
                btn['btnFrame'].config(bg='#e0e0e0')
                btn['imageLabel'].config(bg='#e0e0e0')
                btn['nameLabel'].config(bg='#e0e0e0')

    def clearInstruments(self):
        if globals.instrumentsPlaying:
            commit()
            globals.instrumentsPlaying.clear()
            globals.playbackPosition = 0
        
    def popLast(self):
        if globals.instrumentsPlaying:
            commit()
            globals.instrumentsPlaying.pop()
            
    def undo(self):
        if globals.undoStack:
            globals.redoStack.append([inst for inst in globals.instrumentsPlaying])
            prev = globals.undoStack.pop()
            globals.instrumentsPlaying = [inst for inst in prev]
            
    def redo(self):
        if globals.redoStack:
            globals.undoStack.append([inst for inst in globals.instrumentsPlaying])
            next = globals.redoStack.pop()
            globals.instrumentsPlaying = [inst for inst in next]
            
    def togglePlacement(self):
        globals.isPlacingOn = self.placementVar.get()

    def onVolumeChange(self, value):
        globals.volume = float(value) / 100
    
    def quitApp(self):
        self.root.quit()
    
    def updateFrame(self):
        redrawAll()
        
        imageResized = cv2.resize(globals.image, (self.videoWidth, self.videoHeight))
        imageRGB = cv2.cvtColor(imageResized, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(imageRGB)
        self.photo = ImageTk.PhotoImage(image=pilImage)
        
        self.videoLabel.config(image=self.photo)
        
    def updateInfo(self):
        count = len(globals.instrumentsPlaying)
        self.countLabel.config(text=f"Instruments: {count}")
        
        if globals.isHandVisible:
            if globals.areFingersTouching and globals.isPlacingOn:
                self.statusLabel.config(text="Placing instrument...", fg="#ff8800")
            else:
                self.statusLabel.config(text="Hand detected", fg="#00ff37")
        else:
            self.statusLabel.config(text="Hand not detected", fg='#666666')

def redrawAll():
    fingers = globals.trackingPositions[3]
    indexTip = fingers[8]
    
    # Index tip preview
    if globals.isPlacingOn and globals.isHandVisible:
        currentName = os.path.splitext(os.path.basename(globals.currentInstrument))[0]
        spritePath = os.path.join(globals.IMAGES_DIR, f"{currentName}.png")
        sprite = cv2.imread(spritePath, cv2.IMREAD_UNCHANGED)
        
        if sprite is not None:
            newWidth = int(sprite.shape[1] * indexTip.size)
            newHeight = int(sprite.shape[0] * indexTip.size)
            
            if newWidth > 0 and newHeight > 0:
                scaledSprite = cv2.resize(sprite, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

                x = indexTip.pixelX - newWidth // 2
                y = indexTip.pixelY - newHeight // 2

                # Used AI to help deal with the sprite clipping at edges and transparency
                sprite_x_start = max(0, -x)
                sprite_y_start = max(0, -y)
                sprite_x_end = min(newWidth, globals.image.shape[1] - x)
                sprite_y_end = min(newHeight, globals.image.shape[0] - y)
                
                img_x_start = max(0, x)
                img_y_start = max(0, y)
                img_x_end = min(globals.image.shape[1], x + newWidth)
                img_y_end = min(globals.image.shape[0], y + newHeight)
                
                # Only draw if there's overlap
                if sprite_x_end > sprite_x_start and sprite_y_end > sprite_y_start:
                    clippedSprite = scaledSprite[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]

                    if clippedSprite.shape[2] == 4:
                        alpha = clippedSprite[:, :, 3] / 255.0
                        for c in range(3):
                            globals.image[img_y_start:img_y_end, img_x_start:img_x_end, c] = \
                                alpha * clippedSprite[:, :, c] + (1 - alpha) * globals.image[img_y_start:img_y_end, img_x_start:img_x_end, c]
                    else:
                        globals.image[img_y_start:img_y_end, img_x_start:img_x_end] = clippedSprite[:, :, :3]
    
    # Draw placed instruments sorted by depth
    depthSortedInstrumentsPlaying = sorted(globals.instrumentsPlaying, key=lambda x: x.size)
    for instrument in depthSortedInstrumentsPlaying:
        currentName = os.path.splitext(os.path.basename(instrument.wav))[0]
        spritePath = os.path.join(globals.IMAGES_DIR, f"{currentName}.png")
        sprite = cv2.imread(spritePath, cv2.IMREAD_UNCHANGED)
        
        if sprite is not None:
            newWidth = int(sprite.shape[1] * instrument.size)
            newHeight = int(sprite.shape[0] * instrument.size)
            
            if newWidth > 0 and newHeight > 0:
                scaledSprite = cv2.resize(sprite, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
                
                x = instrument.pixelX - newWidth // 2
                y = instrument.pixelY - newHeight // 2
                
                # Used AI to help deal with the sprite clipping at edges and transparency
                sprite_x_start = max(0, -x)
                sprite_y_start = max(0, -y)
                sprite_x_end = min(newWidth, globals.image.shape[1] - x)
                sprite_y_end = min(newHeight, globals.image.shape[0] - y)
                
                img_x_start = max(0, x)
                img_y_start = max(0, y)
                img_x_end = min(globals.image.shape[1], x + newWidth)
                img_y_end = min(globals.image.shape[0], y + newHeight)
                
                # Only draw if there's overlap
                if sprite_x_end > sprite_x_start and sprite_y_end > sprite_y_start:
                    clippedSprite = scaledSprite[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
                    
                    if clippedSprite.shape[2] == 4:
                        alpha = clippedSprite[:, :, 3] / 255.0
                        for c in range(3):
                            globals.image[img_y_start:img_y_end, img_x_start:img_x_end, c] = \
                                alpha * clippedSprite[:, :, c] + (1 - alpha) * globals.image[img_y_start:img_y_end, img_x_start:img_x_end, c]
                    else:
                        globals.image[img_y_start:img_y_end, img_x_start:img_x_end] = clippedSprite[:, :, :3]