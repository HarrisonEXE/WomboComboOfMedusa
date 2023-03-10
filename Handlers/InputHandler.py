import time
from Classes.OldPhrase import Phrase


timeout = 5  # seconds


def getManualInput():
    print('Enter 5 numbers (1-5)')

    phrase = Phrase()
    i, baseTime = 0, 0
    while i < 5:
        value = input('>')

        # Input validation
        if value not in ["1", "2", "3", "4", "5"]:
            print('Nahhhh, only enter numbers between 1 - 5 (inclusive):')
            phrase = Phrase()
            i, baseTime = 0, 0
            continue

        # Get the rythym by pulling the timing of the inputs
        onset = 0 if i == 0 else time.time() - baseTime
        baseTime = time.time()

        # Check for timeout
        if onset > timeout:
            print("You gotta be quicker than that. Try again!")
            phrase = Phrase()
            i, baseTime = 0, 0
            continue

        # TODO: Maybe make this not a class.
        phrase.append(int(value), onset)
        i += 1

    return phrase
