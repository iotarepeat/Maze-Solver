from mazeSolve import MazeSolve
import cv2, numpy as np
import time
import sys


def display(img, delay=1):
    cv2.imshow("Frame", img)
    k = cv2.waitKey(delay)
    if k == 27:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    s = MazeSolve(sys.argv[1])
    start = time.time()
    image = s.Astar()
    print("Elapsed:", time.time() - start)
    # display(image, 0)
