# Final Exercise - CMPUT 412

This repository contains implementation solutions for exercise 5. For information about the project, please read the report at:

[Celina Sheng's site](https://sites.google.com/ualberta.ca/csheng2-cmput-412/exercise-5) or [Sharyat Singh Bhanwala's Site](https://sites.google.com/ualberta.ca/projects/exercise-5)

## Structure:

There are two packages for this project. Lane_follow and duckiebot_detection. We will discuss the purpose of the python source files for each package (which are located inside the packages src folder).

### 1. Lane_follow

This package contains the code for all three stages. It implements a node that helps the robot drive autonomously in Duckitown. Which includes, follwoing road rules, avoiding collision with citizens of Duckitown, checking in on fellow duckibots when they are down and also parking autonomously in the correct stalls. 

### 2. Duckibot_detection

This package contains code that helps us identify duckie bots that are ahead of us. It implements computer vision and uses the sticker on the back of other bots to detect bots and the distance from the camera. This node is used in the stage two deliverable. 


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).

## Execution:

To run the program, ensure that the variable `$BOT` stores your robot's host name (ie. `csc229xx`), and run the following commands:

```
dts devel build -f # builds locally
dts devel build -f -H $BOT.local # builds on the robot
dts devel run -R $BOT
```

In another terminal, run:
```
dts devel run -H $BOT.local # runs locally and on robot
```

Note: this way of running assumes that your hostname for your local docker does not start with csc229. If it does, you will have to edit the default.sh file to ensure that the lane following node is launched on the robot, and the MLP node is launched locally.

To shutdown the program, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the duckietown template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/duckietown/template-basic).

Build on top of by Celina Sheng, and Sharyat Singh Bhanwala.

Autonomous lane following code was borrowed from Justin Francis.
Duckibot detection code was borrowed from Shaw

Code was also borrowed (and cited in-code) from the following sources:

Justin Francis. We used his template for our lane follow code, which was taken from eclass under the exercise 4 "Lane Follow Package" link. https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
Zepeng Xiao. We used his Duckiebot detection node code, which was taken from GitHub https://github.com/XZPshaw/CMPUT412503_exercise4
