from mazeSolve import MazeSolve, display
import cv2, numpy as np
import time
import sys


if __name__ == "__main__":
    s = MazeSolve(sys.argv[1])
    start = time.time()
    image = s.meetInMiddle()
    print("Elapsed:", time.time() - start)
    # display(image, 0)
