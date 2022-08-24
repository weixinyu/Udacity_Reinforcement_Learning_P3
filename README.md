Documentation
-------------

## Project Environment:
This project is to train two agents control rackets to bounce a ball over a net.

**State Space**
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. So there are 24 variables in total. Each agent receives its own, local observation.

**Action Space**
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

**When Solved?**
 
The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

## Dependencies:
* If you haven't already, please follow the instructions in the DRLND GitHub repository(https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

* Download the Unity Environment(Version 2)from the following links, place the files in the project2/ folder and unzip the file:
	Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
	Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
	Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
	Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

## How To Run the Project:
* Open Continuous_Control.ipynb and follow the instructions;
    - Install needed packages
    - Import the needed modules
    - Create the environment
    - Get the default brain
    - Examine the state and action spaces
    - Try a random action
    - Create the agent
    - Train the model and save the one that solve the problem
    - Plot the score per episode



