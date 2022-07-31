# Surgical Gestures Recognition

Final project in Surgical Data Science course.

Surgical gesture recognition is a common task based on various data, mainly video and kinematics that are recorded in the operating room from surgeons or as part of simulations in order to obtain synthetic data for experiments. In this project we wanted to check the correlation between temporal resolution of the data and the network performance. We found that the network is quite indifferent to integration of several frame rates. Consequently, we had checked hybrid networks that involve both 1D convolutions and RNNs which showed no significant improvement.

![Da Vinci Surgical Robot (2015)](https://upload.wikimedia.org/wikipedia/commons/2/23/Cmglee_Cambridge_Science_Festival_2015_da_Vinci.jpg)
