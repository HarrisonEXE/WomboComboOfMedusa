"""
Adapted By: Harrison Melton
Original Author: Raghavasimhan Sankaranarayanan
Date: 25/02/2023
"""
import pyaudio


class AudioDevice:
    # AudioBox USB 96: Audio (hw:2,0)
    # Line (3- AudioBox USB 96)
    # Analogue 1 + 2 (2- Focusrite USB Audio)
    def __init__(self, listener, rate=48000, frame_size=2400, input_dev_name='Line (3- AudioBox USB 96)', channels=1):
        self.input_device_id = -1
        self.listener = listener
        self.p = pyaudio.PyAudio()
        self._get_dev_id(input_dev_name)

        if self.input_device_id < 0:
            raise AssertionError("Input device not found")

        self.stream = self.p.open(rate=rate, channels=channels, format=self.p.get_format_from_width(2), input=True,
                                  output=False,
                                  frames_per_buffer=frame_size,
                                  stream_callback=self._listener, input_device_index=self.input_device_id)

    def _get_dev_id(self, input_device_name):
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(info)
            if info['name'] == input_device_name:
                self.input_device_id = i

    def _listener(self, in_data, frame_count, time_info, status):
        return self.listener(in_data=in_data, frame_count=frame_count, time_info=time_info, status=status)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def reset(self):
        self.stop()
        if self.stream:
            self.stream.close()
        self.p.terminate()
