import time
from sys import stdout
from flask import Flask, render_template, session, request, jsonify, request, \
    copy_current_request_context, send_from_directory
import logging
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import cv2
import numpy as np
import base64
import io
from PIL import Image
from threading import Lock
from io import StringIO
import threading
lock = threading.Lock()


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
app.config['threaded'] = True
thread = None
socketio = SocketIO(app, async_mode=thread)
thread_lock = Lock()


global fps, prev_recv_time, cnt, fps_array
fps = 30
prev_recv_time = 0
cnt = 0
fps_array = [0]

background_thread = None
# background_lock = Lock()


net = cv2.dnn.readNetFromCaffe(
    './saved_model/MobileNetSSD_deploy.prototxt.txt',
    './saved_model/MobileNetSSD_deploy.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "monitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


@app.route("/")
def index():
    return render_template('index.html',async_mode=socketio.async_mode)


@socketio.on('catch-frame')
def catch_frame(data_image):
    # time.sleep(1)
    global fps, cnt, prev_recv_time, fps_array
    recv_time = time.time()
    # text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))
    #sending frame to detect object from frame
    detect_object(frame)

    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    # print(fps_array)
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0


def detect_object(frame):
    global net
    # print('Calling Object Detect!')
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    confidence = detections[0, 0, 0, 2]

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            try:
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                # print(label)
                # stringlength = len(label)  # calculate length of the list
                # slicedString = label[stringlength::-1]  # slicing
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                #label = np.array(np.rot180(label, 3))
                cv2.putText(frame, label,  (startX, startY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)
                # put it right side up
                imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

                # base64 encode
                stringData = base64.b64encode(imgencode).decode('utf-8')
                b64_src = 'data:image/jpeg;base64,'
                stringData = b64_src + stringData

                # emit the frame back
                
            except Exception as e:
                pass
            emit('response_back', stringData)


def generate():
    # grab global references to the lock variable
    global lock
    global net
    print('Generate Recognization')
    # initialize the video stream
    vc = cv2.VideoCapture(0)

    # check camera is open
    if vc.isOpened():
        vc.set(1, 100)
        rval, frame = vc.read()
    else:
        rval = False

    # while streaming
    while rval:
        # wait until the lock is acquired
        with lock:
            # read next frame
            rval, frame = vc.read()
            # if blank frame
            if frame is None:
                continue

            # video recognization start here
            print('Inside recognization')
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            confidence = detections[0, 0, 0, 2]
            print('after confidence')
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                print('for loop prediction')
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
                if confidence > 0.5:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    try:
                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        # print(label)
                        # stringlength = len(label)  # calculate length of the list
                        # slicedString = label[stringlength::-1]  # slicing
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      COLORS[idx], 1)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        #label = np.array(np.rot180(label, 3))
                        cv2.putText(frame, label,  (startX, startY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)
                        # put it right side up

                    except Exception as e:
                        pass
                    # encode the frame in JPEG format
                    (flag, encodedImage) = cv2.imencode(".jpg", frame)

                    # ensure the frame was successfully encoded
                    if not flag:
                        continue

                    # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        # release the camera

    vc.release()


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})


@socketio.event
def my_event(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})


@socketio.event
def my_broadcast_event(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         broadcast=True)


@socketio.event
def join(message):
    join_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.event
def leave(message):
    leave_room(message['room'])
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'In rooms: ' + ', '.join(rooms()),
          'count': session['receive_count']})


@socketio.on('close_room')
def on_close_room(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response', {'data': 'Room ' + message['room'] + ' is closing.',
                         'count': session['receive_count']},
         to=message['room'])
    close_room(message['room'])


@socketio.event
def my_room_event(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         to=message['room'])


@socketio.event
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)


@socketio.event
def my_ping():
    emit('my_pong')


@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)


if __name__ == '__main__':
    socketio.run(app)