import random
from Classes.Phrase import Phrase

# note length is an eighth note at 60 bpm (.5sec/beat)
note_length = .5


def basic_randomizer(phrase):
    print("randomizing notes...")
    notes = phrase.notes
    random_phrase = Phrase()
    ### for now i am assuming 4/4 measure of eighth notes so 2 measures of 8 notes being played ###
    for i in range(16):
        random_num = random.randint(0, 5)

        #### random_num: 0-4 represent notes being played by user and 5 represents a rest ####
        random_phrase.append(0, note_length) if random_num > 4 else random_phrase.append(
            notes[random_num], note_length)


# note length is an eighth note at 60 bpm (.5sec/beat)
note_length = .5


def basic_randomizer(phrase):
    print("randomizing notes...")
    notes = phrase.notes
    random_phrase = Phrase()
    ### for now i am assuming 4/4 measure of eighth notes so 2 measures of 8 notes being played ###
    for i in range(16):
        random_num = random.randint(0, 5)

        #### random_num: 0-4 represent notes being played by user and 5 represents a rest ####
        random_phrase.append(0, note_length) if random_num > 4 else random_phrase.append(
            notes[random_num], note_length)

    print(f"PHRASE {random_phrase}")
    return random_phrase


def beat_randomizer(phrase):
    print("randomizing beat...")
    random_phrase = Phrase()
    notes = phrase.notes
    length = len(notes)
    random_arr = [1, 1, 1, 2, 2, 2, 2, 2, 2]
    rand_len = len(random_arr) - 1
    for i in range(0, 32, 2):
        # random delay is a random amount of beats
        random_num = random_arr[random.randint(0, rand_len)]
        random_delay = note_length * random_num
        random_phrase.append(notes[i % length], random_delay)
        if (random_num == 1):
            eighth_or_dotted = random.randint(0, 1)
            random_phrase.append(notes[(i + 1) % length], 4 * note_length -
                                 random_delay) if eighth_or_dotted == 1 else random_phrase.append(notes[(i + 1) % length], random_delay)

    return random_phrase


def new_randomizer(phrase):
    print("testing the new randomizer")
    notes = phrase.notes
    onsets = phrase.onsets
    random_phrase = Phrase()
    for i in range(0, 8):
        note_selector = random.randint(0, 4)
        onset_selector = random.randint(0, 4)
        if (onsets[onset_selector] < .005):
            random_phrase.append(notes[note_selector], .3)
        else:
            random_phrase.append(notes[note_selector], onsets[onset_selector])
    return random_phrase
