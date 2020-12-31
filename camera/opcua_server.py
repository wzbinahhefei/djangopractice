import sys
import time
import random
from opcua import ua, Server


sys.path.insert(0, "..")


if __name__ == "__main__":

    # setup our server
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")

    # setup our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = server.register_namespace(uri)

    # get Objects node, this is where we should put our nodes
    objects = server.get_objects_node()

    # populating our address space
    myobj = objects.add_object(idx, "Tank4C9")
    status = myobj.add_variable(idx, "Status", 0)
    overheadFlow = myobj.add_variable(idx, "OverheadFlow", 0)
    buttomsFlow = myobj.add_variable(idx, "ButtomsFlow", 0)
    power = myobj.add_variable(idx, "Power", 0)

    # myVar.set_writable()    # Set MyVariable to be writable by clients

    # starting!
    server.start()

    try:
        count = 0
        while True:
            time.sleep(5)
            count = count + 1
            print(count % 4)
            if count % 4 > 0:
                status.set_value(1)
            else:
                status.set_value(0)

            a = random.randint(100, 500)
            print("OverheadFlow:" + str(a))
            overheadFlow.set_value(a)
            buttomsFlow.set_value(random.randint(50, 500))
            power.set_value(random.randint(1000, 5000))
    finally:
        # close connection, remove subcsriptions, etc
        server.stop()
