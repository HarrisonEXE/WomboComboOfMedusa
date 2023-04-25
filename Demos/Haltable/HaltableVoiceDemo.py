import speech_recognition as sr
from Demos.Interfaces.IRobotDemo import IRobotDemo


class HaltableVoiceDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True):
        super().__init__(robotHandler, is_lab_work)
        self.name = "Haltable Voice Recognition Demo"
        # Scarlett 2i2 USB: Audio (hw:4,0)
        # USB  Live camera: Audio (hw:2,0)
        self.micName = "Scarlett 2i2 USB: Audio (hw:2,0)"

    def start(self):
        self.announceStart()
        self.running = True
        self.readyRobots()
        r = sr.Recognizer()
        return self.listen(r)

    def listen(self, r):
        print(sr.Microphone.list_microphone_names())
        micIndex = sr.Microphone.list_microphone_names().index(self.micName)
        mic = sr.Microphone(micIndex)  # set mic

        with mic as source:
            while self.running:
                print("Speak a command.")
                #r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)

                response = {
                    "success": True,
                    "error": None,
                    "transcription": None
                }

                try:
                    response["transcription"] = r.recognize_google(audio)
                except sr.RequestError:
                    response["success"] = False
                    response["error"] = "API unavailable"
                    continue
                except sr.UnknownValueError:
                    response["error"] = "Unable to recognize speech"
                    continue

                text = response["transcription"]
                text = text.lower()

                print(f"Detected phrase: {text}")
                if "wake up" or "up" in text:
                    print("You have angered Medusa")
                    self.robotHandler.turnOnLive()
                    return True

                elif "goodbye medusa" in text:
                    print("Medusa is going to sleep")
                    self.robotHandler.turnOnLive()
                    return True
