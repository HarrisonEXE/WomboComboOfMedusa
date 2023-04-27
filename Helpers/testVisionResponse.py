import VisionResponse
import numpy as np
import csv

if __name__ == "__main__":
    VisionResponse.init()

    swipe_left = VisionResponse.make_traj(3)

    #write swipe_left to a csv file
    with open('c:/Users/kayla/OneDrive/Desktop/Georgia_Tech/VIP/WomboComboOfMedusa/Helpers/test_gesture.csv', 'w') as f:
        csv_writer = csv.writer(f)
        print(len(swipe_left[0]))
        for i in range(len(swipe_left[0])):
            row = [swipe_left[0][i], swipe_left[1][i], swipe_left[2][i], swipe_left[3][i], swipe_left[4][i], swipe_left[5][i], swipe_left[6][i]]
            csv_writer.writerow(row)