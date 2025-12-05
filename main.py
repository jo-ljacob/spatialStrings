import tkinter as tk
from tracking import initializeTracking, updateTracking
from sound import initializeAudio, updateSources
from gui import GUI
import globals

def main():
    root = tk.Tk()
    gui = GUI(root)
    
    initializeTracking()
    updateTracking()
    initializeAudio()
    
    def updateLoop():
        updateTracking()
        updateSources()
        
        gui.updateFrame()
        gui.updateInfo()
        
        root.after(33, updateLoop)

    updateLoop()
    root.mainloop()
    
    globals.stream.stop()
    globals.stream.close()
    globals.cap.release()

if __name__ == "__main__":
    main()