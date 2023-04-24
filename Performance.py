from queue import Queue
from threading import Thread
import time
from Demos.Haltable.HaltableMicDemo import HaltableMicDemo
from Demos.Haltable.HaltableVoiceDemo import HaltableVoiceDemo
from Demos.VisionTrackerDemo import VisionTrackerDemo
from Handlers.RTPMidiHandler import RtpMidi
from pymidi import server

midiQueue = Queue()


class Performance:
    def __init__(self, robotHandler, is_lab_work=True):
        self.robotHandler = robotHandler

        self.voiceDemo = HaltableVoiceDemo(robotHandler, is_lab_work)
        self.micDemo = HaltableMicDemo(
            robotHandler, is_lab_work, robots_already_awake=True)
        self.visionTrackerDemo = VisionTrackerDemo(
            robotHandler, is_lab_work, robots_already_awake=True)

        self.mainThread = Thread()
        self.rtpMidi = RtpMidi("xArms", MyHandler(), 5004)
        self.midiThread = Thread(target=self.rtp_midi.run())

    def runSequence(self):
        # 0:00 ------------------- 1 -
        # Voice Command (Audio)
        # ----------------------------
        mainThread = Thread(target=self.voiceDemo.start)
        mainThread.start()
        mainThread.join()

        # 0:01 - 0:30 ------------ 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        self.micDemo.start()
        mainThread = Thread(target=self.micDemo.runDemoThread)
        mainThread.start()
        midiQueue.get()
        self.micDemo.kill()
        mainThread.join()

        # 0:31 - 3:00 ------------ 3 -
        # Live Tracking then Gestures (Vision)
        # ----------------------------
        mainThread = Thread(target=self.visionTrackerDemo.start)
        mainThread.start()
        midiQueue.get()
        self.visionTrackerDemo.kill()  # TODO: Verify efficacy
        mainThread.join()

        # 3:01 - 3:30 ------------ 4 -
        # Hope Dance (Preprogrammed)
        # ----------------------------

        # END--------------------- 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        # mainThread = Thread(target=self.voiceDemo.start)
        # mainThread.start()
        # mainThread.join()


class MyHandler(server.Handler):

    def on_peer_connected(self, peer):
        # Handler for peer connected
        print('Peer connected: {}'.format(peer))

    def on_peer_disconnected(self, peer):
        # Handler for peer disconnected
        print('Peer disconnected: {}'.format(peer))

    def on_midi_commands(self, peer, command_list):
        # Handler for midi msgs
        for command in command_list:
            chn = command.channel
            if chn == 1:  # this means its channel 2!!!!!
                if command.command == 'note_on':
                    midiQueue.put(1)

            if chn == 2:  # this means its channel 3 !!!!!
                if command.command == 'note_on':
                    print("DRUMMO")
