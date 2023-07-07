import mss
import cv2
import math
import serial
import time
import torch
import numpy as np
import yaml
import win32gui, win32con, win32api, win32ui
import multiprocessing
import seaborn as sn
import tensorrt as trt
from random import *
from multiprocessing    import Process, Queue
from torchvision        import models
from numpy              import random
from PIL                import ExifTags, Image, ImageOps

imgsz = 416
screenWidth = 1920
screenHeight = 1080
XPAD = (screenWidth / 2) - (imgsz / 2)
YPAD = (screenHeight / 2) - (imgsz / 2)

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=1, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def letterbox(im, new_shape=416, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class LoadScreenshots:
    # screenshot dataloader
    def __init__(self, source, img_size=416, stride=32, auto=False, transforms=None):
        # source = [screen_number left top width height] (pixels)
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
    def __iter__(self):
        return self
    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "
        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s

def computer_vision(queue):
    imgsz=416
    model = torch.hub.load('./', 'custom', path='./CustomModel/best.engine', source='local')  # local repo
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadScreenshots("screen 1 752 332 416 416", img_size=imgsz, stride=stride, auto=pt)

    for path, im, im0s, vid_cap, s in dataset:
        pred = model(im, imgsz)
        queue.put(pred)
        #print('\n',pred,'\n')

def mouse_move(queue_m):
    arduino = None
    # Attempt to setup serial communication over com-port
    try:
        arduino = serial.Serial('COM5', 115200)
        print('Arduino: Ready')
    except:
        print('Arduino: Could not open serial port - Now using virtual input')

    print("Mouse Process: Ready")
    while True:
        # Check if there is data waiting in the queue
        try:
            move_data = queue_m.get()
            out_x, out_y, click = move_data[0], move_data[1], move_data[2]
        except:
            print('Empty')
            continue
        # If using arduino, convert cooridnates
        if(arduino):
            if out_x < 0: 
                out_x = out_x + 256 
            if out_y < 0:
                out_y = out_y + 256 
            pax = [int(out_x), int(out_y), int(click)]
            # Send data to microcontroller to move mouse
            arduino.write(pax)          
        else:
            # Move mouse virtually
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(out_x), int(out_y), 0, 0)
            if (click == 1):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                time.sleep(.1)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
                print('Left Click')

# More accurate sleep(Performance hit)
def sleep_time(wt):
    target_time = time.perf_counter() + (wt / 1000)
    # Busy-wait loop until the current time is greater than or equal to the target time
    while time.perf_counter() < target_time:
        pass

# Linear interpolation(i think)
def lerp(wt, ct, x1, y1, start, queue_m):
    x_, y_, t_ = 0, 0, 0
    for i in range(1, int(ct) + 1):
        xI = i * x1 // ct
        yI = i * y1 // ct
        tI = (i * ct) // ct
        # Put mouse input in queue
        queue_m.put([xI - x_, yI - y_, 0])
        sleep_time(tI - t_)
        x_ = xI
        y_ = yI
        t_ = tI
    # Find time remaining (Wait time - Control time loop)
    loop_time = (time.perf_counter() - start) * 1000
    sleep_time(wt - loop_time)

def aim_loop(queue,queue_m):
    aim_cone = 30
    aimToggle = False
    awp_toggle = False
    mPosX, mPosY = screenWidth/2, screenHeight/2

    print('AimLoop: Ready')
    while True:
        try:
            item = queue.get()
        except Empty:
            print('AimLoop: gave up waiting...')
            continue
        start = time.perf_counter()
        closest_head = aim_cone
        moveByX = 0
        moveByY = 0
        largest = 0
        x1s, x2s, y1s, y2s = 0, 0, 0, 0
        distance = 0
        horizontal_d = 0
        verticle_d = 0
        closest_x = aim_cone
        closest_y = aim_cone
        for *xyxy, conf, cls in item.xyxy[0]:
            x1, x2, y1, y2 = xyxy[0].tolist() + XPAD, xyxy[2].tolist() + XPAD, xyxy[1].tolist() + YPAD, xyxy[3].tolist() + YPAD
            xC, yC = (x1 + x2) / 2 , (y1 + y2) / 2
            cls_ = cls.tolist()
            distance = math.sqrt((xC - mPosX)**2 + (yC - mPosY)**2)
            horizontal_d = abs(xC - mPosX)
            verticle_d = abs(yC - mPosY)
            size = (x2 - x1) * (y2 - y1)
            if(cls_ == 1 or cls_ == 3):
                if(distance < aim_cone and distance < closest_head and size > largest or (awp_toggle and horizontal_d < aim_cone and size > largest)):
                    largest = size
                    closest_head = distance
                    moveByX = xC - mPosX
                    moveByY = yC - mPosY

        if(aimToggle and (moveByX != 0 or moveByY != 0)):
            if (awp_toggle):
                queue_m.put([moveByX/2, 0, 0])
            else:
                #queue_m.put([moveByX/2, moveByY/2, 0])
                queue_m.put([moveByX/1.5, moveByY/1.5, 0])
            #lerp(2, 2, moveByX/1.5, moveByY/1.5, start, queue_m)
            #lerp(2, 2, moveByX/2.5, 0, start, queue_m)
        if(win32api.GetAsyncKeyState(38)):
            aim_cone += 5
            print("Aim Cone now: ", aim_cone,"px")
            time.sleep(.1)
        if(win32api.GetAsyncKeyState(40)):
            aim_cone -= 5
            print("Aim Cone now: ", aim_cone,"px")
            time.sleep(.1)
        if(win32api.GetAsyncKeyState(5)):
            if(aimToggle):
                aimToggle = False
                print("Aimbot Off")
                time.sleep(.2)
            else:
                aimToggle = True
                print("Aimbot On")
                time.sleep(.2)
        if(win32api.GetAsyncKeyState(0x58)):
            if(awp_toggle):
                awp_toggle = False
                print("awp_toggle Off")
                time.sleep(.2)
            else:
                awp_toggle = True
                print("awp_toggle On")
                time.sleep(.2)
            

if __name__ == '__main__':
    multiprocessing.freeze_support()

    queue = Queue(1)
    # Queue to communicate mouse movements, Recoil -> queue_m -> Mouse
    queue_m = Queue(1)

    # Start Mouse process, handles sending mouse out data, communicate with using queue_m
    mouse = Process(target=mouse_move, args=(queue_m,))
    mouse.daemon = True
    mouse.start()

    consumer = Process(target=aim_loop, args=(queue,queue_m,))
    consumer.daemon = True
    consumer.start()

    computer_vision(queue)

