math-grader <br>
[![Build Status](https://travis-ci.com/mammothb/math-grader.svg?branch=master)](https://travis-ci.com/mammothb/math-grader)
[![codecov](https://codecov.io/gh/mammothb/math-grader/branch/master/graph/badge.svg?token=Q2CD0IVP7B)](https://codecov.io/gh/mammothb/math-grader)
========
This repository contains the code for a automated math worksheet grader, implemented as part of the requirements of the "deep skilling" phase of AI
Singapore's apprenticeship programme.

I have performed further refinements of the codebase such as:
- Reorganizing/modularizing various classes and functions
- Implementing unit tests to achieve 100% code coverage of backend code
- Set up Travis CI and codecov for code health report
- Host the app ([link](https://share.streamlit.io/mammothb/math-grader))

### Problem statement:
Marking of handwritten math homework can be tedious and laborious, hence it is prone to human error.

### Our proposed solution:
Automate marking of (simple) math equations on homework submitted

# Workflow
1. Upload a photo of the worksheet (with all edges of the worksheet visible)
2. After a series of image manipulations, the equations are extracted from the worksheet (as rectangular images)
3. For each equation, the digits/symbols are extracted and fed to the classification model
4. The classification results are joined and evaluated to mark the equations.
5. For each equation on the extracted worksheet, a green box is drawn if the equation is correct and a red box is drawn if the equation is wrong.

# Model training
The following model architecture is used
```
[[Conv2D->ReLU] * 2 -> MaxPool2D -> Dropout] * 2 -> Flatten -> Dense -> Dropout -> Out
```
A combination of the [Handwritten math symbols](https://www.kaggle.com/xainano/handwrittenmathsymbols) and MNIST dataset was used to train the model.

# Contributors
**AIAP Batch 6**

Mentor: Ng Lee Ping

Members:

- Heng Kwan Wee
- Li Yier
- Mok Kay Yong
- Theodore Lee

# Repository
[GitHub](https://github.com/mammothb/math-grader)
