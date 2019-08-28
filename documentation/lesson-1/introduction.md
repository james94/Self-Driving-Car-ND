# Introduction

## Why Self-Driving Cars?

Self-Driving cars can drive safer than humans

## Overview of Self-Driving Car Development

Self-Driving Car Engineers use Robotics or Deep Learning as an approach to autonomous development.

The Robotics Approach fuses output from a suite sensors to directly measure the vehicle surroundings and then navigate accordingly.

The Deep Learning Approach uses deep neural networks to allow cars to learn how to drive by mimicking human driving behavior.

Companies:

- Ford
- Mercedes
- OTTO
- AS
- Nvidia
- Uber

You can also develop both Robotics and Deep Learning code to control a real self-driving car.

Work through the modules with each layer focusing on a different layer of the self-driving car technology stack

![self-driving car tech stack]()

## What Projects Will You Build?

The projects you will be added to the self-

## Layer 1: Computer Vision

### Project 1: Finding Lane Lines

The first project you will build for the self-driving car technology stack is one to **find lane markings** by using fundamental computer vision techniques. If you can't find lane markings you probably won't know where to drive. As a Self-Driving Car Engineer, they are very helpful since they tell you a lot as where to drive.

### Project 2: Advanced Lane Finding

The second project you will build for the self-driving car technology stack enhances the first project by using advanced computer vision techniques. What is computer vision? When humans drive, they use their eyes more than any other organ to understand what to do. Computer Vision does this same capability in computers. It uses cameras and we learn how to extract things like lane markings and other vehicles out of camera images.

## Layer 2: Deep Learning

### Project 3: Traffic Sign Classifier

The third project you will train a convolutional neural network based on the LeNet neural network architcture to recognize traffic signs and images.

### Project 4: Behavioral Cloning

The fourth project you will train a convolutional neural network based on Nvidia's Self Driving Car neural network architecture to clone human behavior such as our steering actions, brake and gas actions.

## Layer 3: Sensor Fusion

### Project 5: Extended Kalman Filter

The fith project you will use sensor fusion to integrate different types of sensors (like LiDAR, RADAR, cameras, gyroscoppes, inertial measurement units) and come up with a clear picture based on these different modalities.

## Layer 4: Localization

### Project 6: Kidnapped Vehicle

Localization is the question where relative to the world is a car. You are going to build maps of the environment with lots of detailed information some of which might even be invisible to the robot. Like where are the danger zones? Where do pedestrians typically cross? To make sense of those maps, the robot must know where it is right from the map. The alignment of the sensor data, the current position to a map is what is called Localization.

## Layer 5: Path Planning

### Project 7: Path Planning Highway Driving

Path Planning is how to find a valid sequence of steps or actions in a maze. But even in Southern Manhattan for example, you want to cross the city, and you have to find a plan, a sequence of streeets you will go down to make it all the way to the other end. A path planning algorithm is an algorithm that finds that sequence of actions for you. It can apply to city navigation, parking lot navigation. 

## Layer 6: PID Control

### Project 8: PID Controller

A PID Controller is a computer program that steers your wheels, your gas pedal, your brake pedal so as to meet a given objective. So if your objective is to stay centered in the road and you start driving and your car veers a little to the left, your controller is the unit to decide, okay we compensate with the slight right under the steering wheel.

## Layer 7: System Integration

### Project 9: Put Your Code In A Real Self-Driving Car

Take your code, put it onto a physical car and drive it around in California.