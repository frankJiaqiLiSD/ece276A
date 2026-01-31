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
#show heading: set text(weight: "regular", size: 15pt)


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
= Introduction
Orientation tracking and pose estimation are the bases for almost all robotic problems in the autonomous robot industry. Whether for drones or autonomous vehicles, the ability to accurately determine its current state is essential for effective control and navigation. 

Inertial Measurement Unit (IMU) is one of the mostly used sensors in the robotics field. Its affordability and robustness has made it widely used in pose estimation, which is one of the key problems that robotic engineers deal with when designing autonomous moving algorithms. 

Typical IMU sensors consists of three accelerometers and three gyroscopes to measure the linear acceleations and angular velocities with respect to their corresponding axes. In this project, the IMU used track 6 datas at the same time and represent them in the following format:
#align(center)[#table(columns: (auto,1fr,1fr,1fr,1fr,1fr,1fr), [Unix Time Stamps],[$"A"_"x"$],[$"A"_"y"$],[$"A"_"z"$],[$"W"_"x"$],[$"W"_"y"$],[$"W"_"z"$])]


]