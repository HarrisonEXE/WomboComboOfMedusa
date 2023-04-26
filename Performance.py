from queue import Queue
from threading import Thread
import time
from Demos.Haltable.HaltableMicDemo import HaltableMicDemo
from Demos.Haltable.HaltableVoiceDemo import HaltableVoiceDemo
from Demos.VisionTrackerDemo import VisionTrackerDemo
from pymidi import server

from Handlers.RTPMidiHandler import RtpMidi

midiQueue = Queue()

class Performance:
    def __init__(self, robotHandler, is_lab_work=True):
        self.robotHandler = robotHandler

        self.voiceDemo = HaltableVoiceDemo(robotHandler, is_lab_work)
        self.micDemo = HaltableMicDemo(robotHandler, is_lab_work, robots_already_awake=True)
        # self.visionTrackerDemo = VisionTrackerDemo(robotHandler, is_lab_work, robots_already_awake=True)
        self.visionTrackerDemo = VisionTrackerDemo(robotHandler, is_lab_work, robots_already_awake=False)

        self.mainThread = Thread()
        self.rtp_midi = RtpMidi("xArms", MyHandler(self.robotHandler), 5004)
        self.midiThread = Thread(target=self.rtp_midi.run)

    def runSequence(self):
        print("starting thread")
        self.midiThread.start()
        print("running ")

        # 0:00 ------------------- 1 -
        # Voice Command (Audio)
        # ----------------------------
        # mainThread = Thread(target=self.voiceDemo.start)
        # mainThread.start()
        # mainThread.join()

        # 0:01 - 0:30 ------------ 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        # self.micDemo.start()
        # mainThread = Thread(target=self.micDemo.runDemoThread)
        # mainThread.start()
        # # TODO: Kill based on r2pmidi, maybe improvise
        # midiQueue.get()
        # self.micDemo.kill()
        # mainThread.join()

        # TODO: Audio blends into presequence

        # 0:31 - 3:30 ------------ 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        mainThread = Thread(target=self.visionTrackerDemo.start)
        mainThread.start()
        # TODO: Figure out transition to end

        # END--------------------- 2 -
        # Xylophone Mimicking (Audio)
        # # ----------------------------
        # mainThread = Thread(target=self.voiceDemo.start)
        # mainThread.start()
        # mainThread.join()


class MyHandler(server.Handler):
    def __init__(self, robotHandler):
        self.robotHandler = robotHandler

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
            if chn == 2:  # this means its channel 3 !!!!!
                if command.command == 'note_on':
                    print("Channel Two Note")
                    # midiQueue.put(1)

            if chn == 3:  # this means its channel 4 !!!!!
                if command.command == 'note_on':
                    print("Drum Note Recognized")
                    self.robotHandler.switch_drum_state()
                    print("midi note - play/pause drums")
                    #midiQueue.put(1)

