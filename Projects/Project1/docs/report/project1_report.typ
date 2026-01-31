// Setting up the page
#set page(paper: "us-letter", 
          margin: (top: 0.75in, 
                   bottom: 1in, 
                   left: 0.625in, 
                   right: 0.625in),
          numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "I.")
#show heading: set align(center)
#show heading: set text(weight: "regular", size: 10pt)
#set math.equation(numbering: "(1)")


// Formatting Titles
#let make_title(title) = {
  align(center)[
    #text(size: 20pt, weight: "bold", font: "New Computer Modern", title)
  ]
}

// Formatting Authors
#let make_authors() = {
  grid(columns: (1fr, 1fr),align: center, 
  [*Jiaqi Li*\ 
  _Department of Electrical and Computer Engineering_\
  _University of California, San Diego_\
  La Jolla, U.S.A.\
  jil547$@$ucsd.edu],
  [*Nikolay Atanasov*\ 
  _Department of Electrical and Computer Engineering_\
  _University of California, San Diego_\
  La Jolla, U.S.A.\
  natanasov$@$ucsd.edu])
  v(2em)
}

// Formatting abstract and Index Terms
#let abstract_index(abstract, index) = {
  text(weight: "bold")[#h(1em) _Abstract_ - #abstract\ #h(1em) _Index Terms_ - #index]
}

// ------------- Content Begin -------------
#make_title("IMU-based Orientation Estimation and Trajectory Optimization")
#make_authors()

// Abstrac and Index Terms
#columns(2)[
#abstract_index("This project focuses on tracking the orientation of the camera system built. In this project, angular velocities and accelerations captured by the IMU were first converted to quaternions representing the object's pose, estimating the pose in the next timestamp, and finally use gradient descent algorithm to minimize the difference of its estimations with the data gathered by the VICON camera. The optimized quaternion data was then used in generating panorama pictures to represent the rotations of the camera", "Orientation tracking, Projected gradient descent, Inertial Measurement Unit, Optimization")
= INTRODUCTION
#v(0.5em)
Orientation tracking and pose estimation are the bases for almost all robotic problems in the autonomous robot industry. Whether for drones or autonomous vehicles, the ability to accurately determine its current state is essential for effective control and navigation. In this project, we try to the challenge of estimating the 3-D orientation of a rigid body using noisy measurements from an Inertial Measurement Unit (IMU).

Inertial Measurement Unit (IMU) is one of the mostly used sensors in the robotics field. Its affordability and robustness has made it widely used in pose estimations. Typical IMU sensors consists of three accelerometers and three gyroscopes to measure the linear acceleations and angular velocities with respect to their corresponding axes. In this project, the IMU used track 6 datas at the same time and represent them in the following format:
#align(center)[#table(columns: (auto,1fr,1fr,1fr,1fr,1fr,1fr), [Unix Time Stamps],[$"A"_"x"$],[$"A"_"y"$],[$"A"_"z"$],[$"W"_"x"$],[$"W"_"y"$],[$"W"_"z"$])]

With the provided data, we can get the converted quaternion to utilize the motion model and observation model for estimations. We implemented a projected gradient descent algorithm to minimize a cost function, while strictly enforcing the unit-norm constraints. Finally, to validate the accuracy of our orientation estimates, we construct a panoramic image by stitching together RGB camera frames captured by the rotating body.
#figure(
  image("/assets/flowchart-2.png", width: 70%),
  caption: [Orientation Tracking and Optimization Flowchart])
= PROBLEM FORMATION
#v(0.5em)
In the case that the rigid bosy is going through pure rotation, we can represent it orientation (or pose) in multiple ways: Euler Angle, Axis-rotation, and quaternion. In order to avoid Gimbal Lock and keeping stability in optimization, we use unit quaternions $bold(q)_t in HH_*$  to represent pose in this porject. 

When we are trying to estimate the most accurate orientation of the rigid object, we are actually trying to minimize the difference between the predicted value and the true value. Or, in other words, we are trying to minimize the cost function $c(bold(q_"1:T"))$. We can formulate it as follows:
$ min_(bold(q)_(1:T))" "c(bold(q)_(1:T)) $
The error of the prediction comes from two parts: the difference between the estimated orientation and the motion model, and the difference between the acceleration and the predicted observation model. 

In a 3D world, for every timestamp, we receive angular velocity vector $bold(omega)_t in RR^3$ and linear acceleration velocity vector $bold(a)_t in RR^3$. A motion model is used to estimate the orientation directly from the measured angular velocities. Along with the time difference $tau_t$, we can get the following formula to calculate the estimated $bold(q)_(t+1)$:
$ bold(q_(t+1)) = f(bold(q)_t, tau_t bold(omega)_t) = bold(q)_t compose exp([0, tau_t bold(omega)_t/2]) $

The observation model is used to check whether "the truely observed value is align with what expected to observe". By using the observation model, we want to make sure that the observed acceleation agrees with the gravity acceleration after it was transformed to the IMU frame. 
]