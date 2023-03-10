import speech_recognition as sr
from Demos.IRobotDemo import IRobotDemo


class VoiceDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True):
        super().__init__(robotHandler, is_lab_work)
        self.name = "Voice Recognition Demo"

    def start(self):
        self.readyRobotsWithoutLive()
        r = sr.Recognizer()
        self.listen(r)

    def listen(self, r):
        vibin = True
        with sr.Microphone() as source:
            while vibin:
                print("Speak a command.")
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)
                # text = r.recognize_google(audio)

                response = {
                    "success": True,
                    "error": None,
                    "transcription": None
                }

                # try recognizing the speech in the recording
                # if a RequestError or UnknownValueError exception is caught,
                #     update the response object accordingly
                try:
                    response["transcription"] = r.recognize_google(audio)
                except sr.RequestError:
                    # API was unreachable or unresponsive
                    response["success"] = False
                    response["error"] = "API unavailable"
                    continue
                except sr.UnknownValueError:
                    # speech was unintelligible
                    response["error"] = "Unable to recognize speech"
                    continue

                text = response["transcription"]
                text = text.lower()

                print(f"Detected phrase: {text}")
                if "hey medusa" in text:
                    print("You have angered Medusa")
                    self.robotHandler.turnOffLive()
                    self.robotHandler.scare()
                elif "calm down" in text:
                    print("Aight, Medusa is chill now")
                    self.robotHandler.turnOnLive()
                elif "toggle lights" in text:
                    print("lights are toggled")
                    # switchLightMode()
                    self.robotHandler.switchLightMode()
                    self.robotHandler.lightQ.put(3)
                else:
                    print("sucks")
