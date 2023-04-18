import VisionResponse
import numpy as np

if __name__ == "__main__":
    VisionResponse.init()

    swipe_left = VisionResponse.make_traj(3)

    for item in swipe_left:
        print(type(item))
        print(len(item))