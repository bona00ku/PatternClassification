#
# Client in Python
# Connects REQ socket to tcp://localhost:5555
# Sends str type int to server, expects "Received" back
#
# INSTALLED VIA EASY_INSTALL
import zmq

context = zmq.Context()

# Socket to talk to server
print "Connecting to hello world server..."
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Do 10 requests, waiting each time for a response
for request in range(10):
    print "Sending request ", request, "..."
    socket.send(str(request))

    # Get the reply.
    message = socket.recv()
    print "Received reply ", request, "[", message, "]"
