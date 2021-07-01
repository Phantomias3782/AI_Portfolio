"""epuck_collector controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot,Receiver, Motor, Camera
import msgpack
import logging as log
import numpy as np
from webcolors import rgb_to_name
import time
import argparse
import cv2
from scipy import ndimage
from colorthief import ColorThief
# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

rec:Receiver = robot.getDevice("receiver")
rec.enable(timestep)
log.basicConfig(level=log.INFO, format='%(asctime)s %(filename)s %(levelname)s: %(message)s')

camera = robot.getDevice("camera")
camera.enable(timestep)

motorLeft:Motor = robot.getDevice('left wheel motor')
motorRight:Motor = robot.getDevice('right wheel motor')
motorLeft.setPosition(float('inf')) #this sets the motor to velocity control instead of position control
motorRight.setPosition(float('inf'))
motorLeft.setVelocity(0)
motorRight.setVelocity(0)

width = camera.getWidth()
while robot.step(timestep) != -1:

    try:

        # save image as np array
        camera_image = np.array(camera.getImageArray())

        # rotate image
        camera_image = ndimage.rotate(camera_image, 270)

        # flip image
        camera_image = cv2.flip(camera_image, 1)
        camera_image = np.float32(camera_image)
        output = camera_image.copy()
        
        # grayscale image
        gray_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)

        cv2.imwrite("camera_image.jpg", gray_image)
        gray_image = cv2.imread('camera_image.jpg', 0) 
    
        # get circles
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20,
        param1=30, param2=20, minRadius = 0, maxRadius = 100000)
        
        # ensure at least some circles were found
        if circles is not None:
            
            # notice user
            print("detected")
            
            # convert variables to integers
            circles = np.round(circles[0, :]).astype("int")
            
            # extract first circle
            (x, y, r) = circles[0]
            
            # save circle in image
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)
            cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)
            
            # define image center
            center_line = (width/2)
            
            # save crop
            crop = camera_image[y:y+r,x:x+r] 
            cv2.imwrite("crop.jpg", crop)
            color_thief = ColorThief('crop.jpg')
            
            # check if ball is green
            dominant_color = list(color_thief.get_color(quality=1))
            red = dominant_color[0]
            green = dominant_color[1]
            blue = dominant_color[2]
            
            print("rgb: ", red, green, blue)
            
            if green > red and green > blue:
                
                print("green color detected")
                
                # start rotation
                motorRight.setVelocity(6)
                motorLeft.setVelocity(-6)
                
            else:
            
                # if x is in range of x +/- 10% move foreward
                if center_line * 0.9 <= x <= center_line * 1.1:
                
                    print("moving to ball")
                    motorRight.setVelocity(6)
                    motorLeft.setVelocity(6)
                        
                elif x < center_line * 0.9:
                    
                    print("moving left")
                    motorRight.setVelocity(3)
                    motorLeft.setVelocity(0)
                    
                elif x > center_line * 1.1:
                    
                    print("moving right")
                    motorLeft.setVelocity(3)
                    motorRight.setVelocity(0)
    
            # save output image with circles
            cv2.imwrite("comparison.jpg", np.hstack([camera_image, output]))
            cv2.imwrite("circles.jpg", output)
            
        else:
        
            print("nothing")
            # start rotation
            motorLeft.setVelocity(0)
            motorRight.setVelocity(3)

        while rec.getQueueLength() > 0:
            msg_dat = rec.getData()
            rec.nextPacket()
            msg = msgpack.unpackb(msg_dat)
            log.info(msg)

    except:
    
        continue

# Enter here exit cleanup code.
