# flask-stream-object-recognization

This project acts as a boilerplate to utilize when serving realtime videos to client over websockets.

---

### Getting Started

Install required dependencies using:

```bash
pip install -r requirements.txt
```

Start the application using

```git
python application.py
```

This uses flask-socketio, eventlet ( to maximize performance via utilization of websockets ).
_With the usage of eventlet, native threading functionalities don't work. Hence to process images using opencv or any image processing library of your choice. Make changes in the [get_frames.py](./get_frames.py)_

### Implementing ML code
To implement custom features such as object detection and sending metadata along with the video stream.
