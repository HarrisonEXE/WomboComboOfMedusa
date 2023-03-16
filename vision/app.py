from xarm_controller import XArmController
from queue import Queue
from pythonosc import dispatcher, osc_server
from threading import Thread, Event
import atexit

server_ip = "0.0.0.0"
server_port = 12345

def handler(address, *args):
    print("Received OSC message:", address, args)
    if address == "/gestures":
        if args[0] == "Hand Open":
            q0.queue()
        elif args[0] == "Hand Closed":
            q0.queue().clear()


def server(ip, port, msg_dispatcher):
    msg_server = osc_server.ThreadingOSCUDPServer((ip, port), msg_dispatcher)
    print("Serving on {}".format(msg_server.server_address))
    msg_server.serve_forever()
    atexit.register(msg_server.server_close())


if __name__ == '__main__':
    xarms = XArmController()
    input("Press enter to continue setup arms...")
    xarms.setup()
    input("Press enter to continue setup position...")
    xarms.setup_pos()

    q0 = Queue()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    dq0 = Queue()
    dq1 = Queue()

    t0 = Thread(target=xarms.move, args=(q0, xarms.arm0,))
    t1 = Thread(target=xarms.move, args=(q1, xarms.arm1,))
    t2 = Thread(target=xarms.move, args=(q2, xarms.arm2,))
    t3 = Thread(target=xarms.move, args=(q3, xarms.arm3,))
    t4 = Thread(target=xarms.move, args=(q4, xarms.arm4,))
    dt0 = Thread(target=xarms.move, args=(dq0, xarms.arm0,))
    dt1 = Thread(target=xarms.move, args=(dq1, xarms.arm1,))

    input("Press enter to start enable arms queue...")
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    dt0.start()
    dt1.start()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/gestures", handler)
    serve = Thread(target=server, args=(server_ip, server_port, dispatcher,))

    input("Press enter to start listening...")
    serve.start()
