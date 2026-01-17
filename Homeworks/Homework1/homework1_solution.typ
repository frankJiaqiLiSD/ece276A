#set page("us-letter",
          header: [*ECE 276A: Sensing and Estimation in Robotics#h(1fr)Homework 1*],
          footer: context[Jiaqi Li, A69041576#h(1fr)#counter(page).display("1")],
          margin: 10%)
#set math.mat(delim: "[")
= Homework 1
== Problem 1
From the description of the function $f(A)$, we can know that we need to first find the two eiganvalues of the matrix $A$:
$ A = mat(delim: "[", 0, a; -a, 0)\
  det(A-lambda I) = det(mat(delim: "[", 0,a;-a,0)-mat(delim: "[", lambda,0;0,lambda)) &= 0\
  det mat(delim: "[", -lambda, a;-a, lambda) &= 0\ 
  lambda^2 + a^2 &= 0 $
From this polynomial equation, we can find two values for eiganvalues: $lambda_1= -a i, lambda_2 = a i$. By taking them into the below equation, we can find their corresponding eiganvectors:
$ (A-lambda I) = 0 $
when $lambda = -a i$, we have: 
$ mat(delim: "[", a i, a;-a,a i) mat(delim: "[", x_1; x_2) = 0 $
and finally $Q_1 = mat(delim: "[", x_1; X_2) = mat(delim: "[", 1; -i)$. Similarly, if we plug in the second eiganvalue, we can find the second eiganvector $Q_2$: 
$ mat(delim: "[", -a i, a;-a,-a i) mat(delim: "[", x_1; x_2) = 0, Q_2 = mat(delim: "[", 1;i) $
With the given function $f(A)$, we can have:
$ f(A) &= Q f(Lambda)Q^(-1) = Q "diag"(f(lambda_1), f(lambda_2))Q^(-1)\ $
where $Q = mat(Q_1, Q_2) = mat(1,1;-i,i) $. To calculate its inverse, we have:
$ det(Q) = (1 dot i)-(1 dot -i) = 2i\
  "Adj"(Q) = mat(i, -1; i, 1)\
  Q^(-1) = 1/det(Q) "Adj"(Q) = mat(1/2, i/2; 1/2, -i/2) $
Finally, we have:
$ f(A) &= mat(1,1;-i,i) mat(e^(-a i), 0; 0, e^(a i)) mat(1/2, i/2; 1/2, -i/2)\
  &= mat(e^(-a i), e^(a i);-i e^(-a i), i e^(a i)) mat(1/2, i/2; 1/2, -i/2) = mat(1/2 e^(-a i)+1/2 e^(a i), i/2 e^(-a i)+i/2 e^(a i); -i/2 e^(-a i)+i/2 e^(-a i), 1/2 e^(-a i)+1/2 e^(-a i)) $
By the definition of $sin(x)$ and $cos(x)$, we can simplify it and get the final result: 
$ f(A) = mat(cos(a), sin(a); -sin(a), cos(a)) $
== Problem 2
(a) Partial derivative of $f(x,y)$ with respect of $r$ is:
$ (delta f)/(delta r) &= (delta f)/(delta x) dot (delta x)/(delta r) + (delta f)/(delta y) dot (delta y)/(delta r)\
  &= (delta f)/(delta x) dot cos(theta) + (delta f)/(delta y) dot sin(theta)\
  &= f_x dot cos(theta) + f_y dot sin(theta) $
(b) Partial derivative of $f(x,y)$ with respect of $theta$ is:
$ (delta f)/(delta theta) &= (delta f)/(delta x) dot (delta x)/(delta theta) + (delta f)/(delta y) dot (delta y)/(delta theta)\
  &= (delta f)/(delta x) dot -r sin(theta) + (delta f)/(delta y) dot r cos(theta)\
  &= f_x dot -r sin(theta) + f_y dot r cos(theta) $
== Problem 3
*(a)(1)* With the definition of norm $||bold(q)|| = sqrt(bold(q)^T bold(q))$, we can simplify the function $f$ as $f(bold(q)) = 1/2log(bold(q)^T bold(q))$. By chain rule, we can have:
$ (d f)/(d bold(q)) = (delta f)/(delta u) dot (delta u)/(delta bold(q)) $
where $u = bold(q)^T bold(q)$. By calculating each part, we get:
$ (delta u)/(delta bold(q)) &= 2bold(q)\
  (delta f)/(delta u) &= 1/2 dot 1/u ln(e) = 1/(2u)\
  (d f)/(d bold(q)) &= (2bold(q))/(2 ||bold(q)||^2) = (bold(q))/(||bold(q)||^2) $

*(a)(2)* We can first change the format of the function $g$ to the following:
$ g(bold(q)) = (bold(q)_v)/(||bold(q_v)||) = bold(q_v)||bold(q_v)||^(-1) $
Now we implement the product rule to calculate its derivative: 
$ (d g(bold(q)))/(d bold(q)) &= bold(q)_v (d ||bold(q)_v||^(-1))/(d bold(q)) + (d bold(q)_v)/(d bold(q)) ||bold(q)_v||^(-1)\
  &= bold(q)_v U + I||bold(q)_v||^(-1) $
Now we calculate the value for $U$: 
$ U = (d ||bold(q)_v||^(-1))/(d bold(q)), "and we set" A &= ||bold(q)_v||^(-1) = (bold(q)_v^T bold(q)_v)^(-1/2)\
  (d A)/(d bold(q)_v) &= -||bold(q)_v||^(-3)bold(q)_v $

Thus we can have the final result: 
$ (d g(bold(q)))/(d bold(q)) &= - bold(q)_v ||bold(q)_v||^(-3)bold(q)_v + I||bold(q)_v||^(-1)\
  &= - bold(q)_v^T bold(q)_v ||bold(q)_v||^(-3) + I||bold(q)_v||^(-1) $
*(a)(3)* For $arccos(x)$, we have the following formula as its derivative:
$ arccos'(x) = -(1)/(sqrt(1-x^2)) $
By taking our matrix into it, we can have the followings:
$ (d h(bold(q)))/(d bold(q)) = -(1)/(sqrt(1-u^2)), " where " u = q_s ||bold(q)||^(-1) $
By chain rule, we can know that $(d h(bold(q)))/(d bold(q)) = (d h(bold(q)))/(d u) dot (d u)/(d bold(q))$. We have calculates the value for $(d u)/(d bold(q))$ in the previous questions, which gives us:
$ (d h(bold(q)))/(d bold(q)) &= (d h(bold(q)))/(d u) dot (d u)/(d bold(q))\
  &= -(1)/(sqrt(1-u^2)) dot q_s -||bold(q)||^(-3)bold(q)\
  &= (q_s ||bold(q)||^(-3))/(sqrt(1-(q_s ||bold(q)||^(-1))^2)) dot bold(q) $

*(b)* From the below screenshot, we can see that the results matched:
#grid(columns: (45%, 45%), gutter: 10pt,
      image("assets/image-2.png", height: auto, width: 100%),
      image("assets/image-3.png", height: auto, width: 100%))
== Problem 4
In Gauss-Newton method, we need first find the function $e(x) in RR$. In this problem, the best choice is $sin(x)-1/2$. To find its Jacobian matrix, we take its derivative, which is $cos(x)$. Thus, we can form the following equation to find the descent direction, $delta x_k$:
$ delta x_k &= -(J^T J)^(-1)(J^T e(x_k))\
            &= -(cos(x_k)^T cos(x_k))^(-1)(cos(x_k)^T (sin(x_k)-1/2))\
            &= -(cos(x_k)(sin(x_k)-1/2))/(cos(x_k)cos(x_k))\
            &= -(sin(x_k)-1/2)/(cos(x_k)) $
With the fixed and given step size $alpha = 1/2$, we can calculate the following steps:
$ x_0 &= 2\
  x_1 &= 2 - 1/2 times -(sin(2)-1/2)/(cos(2)) = 2.49177\
  x_2 &= 2.49177 - 1/2 times -(sin(2.49177)-1/2)/(cos(2.49177)) = 2.55774 $ 
== Problem 5
This problem is solved largely relying on Python.\
In order to convert the coordinate of $x$ from frame {C} to frame {A}, we need to first convert it to frame {B}. Since the parametrization method used from frame {B} to frame {C} is axix-angle, we frist calculate the rotation matrix in frame {B}: 
$ R &= exp(hat(theta)) $
where $hat(theta) = theta bold(eta) = pi/6 dot mat(1/sqrt(2), 1/sqrt(2), 0)^T = mat(pi/(6sqrt(2)), pi/(6sqrt(2)), 0)^T$. Using Rodrigues formula, we can calculate the rotation matrix in frame {B}:
$ R = I+((sin(||theta||))/(||theta||))hat(theta)+((1-cos(||theta||))/(||theta||^2))hat(theta)^2 $
Using Python, we can get the calculated matrix:
$ R = mat(0.9330,  0.0670,  0.3536;
          0.0670,  0.9330, -0.3536;
          -0.3536,  0.3536,  0.8660) $
We can also calculate the matrix $T$ to get the coordinate of $x$ in frame {B}:
$ ""_{B} T_{C} &= mat(R,P;0^T,1)mat(s_C;1)\
               &= mat(1.3536;1.6464;2.8660) $
Now we get the coordinate of $x$ in frame {B}, we can try to convert it to frame {A}. In this transition, we used Euler-Angle method, which needs to multiply rotation axis metrices: 

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,[
$ R_x(phi) = mat(
  1, 0, 0;
  0, cos(phi), -sin(phi);
  0, sin(phi), cos(phi)
) $],
[$ R_y(theta) = mat(
  cos(theta), 0, sin(theta);
  0, 1, 0;
  -sin(theta), 0, cos(theta)
) $],[
$ R_z(psi) = mat(
  cos(psi), -sin(psi), 0;
  sin(psi), cos(psi), 0;
  0, 0, 1
) $])
Using Python, we can calculate the final coordinateeasily using the following equation:
$ s x y z = r z y x $
and get the final coordinate:
$ ""_{A} T_{B} &= R_z(psi)R_y(theta)R_x(phi)mat(1.3536;1.6464;2.8660) + mat(1;1;0)\
               &= mat(3.5077;3.4976;0.4804) $
#pagebreak()
== Appendix
I have pasted the Python codes used in solving above questions below for checking.\
*Used in Problem 3*\
```python
import torch
def logorithm(q):
    return torch.log(torch.norm(q))
def fraction(qv):
    return (qv/(torch.norm(qv)))
def arccos(q):
    qs = 1.0
    return torch.acos(qs/(torch.norm(q)))
q = torch.tensor([[1.], [2.], [3.], [4.]])

jacobian_log_torch = torch.autograd.functional.jacobian(logorithm, q)
print("PyTorch Version: \n", jacobian_log_torch)

norm_q = torch.norm(q)
jacobian_log_manual = q / torch.pow(norm_q,2)
print("Manual Version: \n", jacobian_log_manual)
qv = torch.tensor([[2.], [3.], [4.]])

jacobian_fraction_torch = torch.autograd.functional.jacobian(fraction, qv)
jacobian_fraction_torch = jacobian_fraction_torch.squeeze()
print("PyTorch Version: \n", jacobian_fraction_torch)

norm_qv = torch.norm(qv)
I = torch.eye(3)
first_term = -qv @ qv.T * torch.pow(norm_qv, -3)
second_term = I*torch.pow(norm_qv, -1)
jacobian_fraction_manual = first_term + second_term
print("Manual Version: \n", jacobian_fraction_manual)
qs = 1.0

jacobian_fraction_torch = torch.autograd.functional.jacobian(arccos, q)
print("PyTorch Version: \n", jacobian_fraction_torch)

norm_q = torch.norm(q)
scaler_part = (qs*torch.pow(norm_q, -3))/(torch.sqrt(1-torch.pow(qs*torch.pow(norm_q, -1),2)))
jacobian_fraction_manual = scaler_part*q
print("Manual Version: \n", jacobian_fraction_manual)
```
*Used in Problem 4*\
```python
import math

x_lst = [2]
for x in range(1,11):
    xk = x_lst[x-1]
    delta = -(math.sin(xk)-0.5)/(math.cos(xk))
    xkp1 = xk + 0.5 * delta
    x_lst.append(xkp1)

print(x_lst)
```
*Used in Problem 5*\
```python
import math
vector_theta = torch.tensor([[(math.pi)/(6*math.sqrt(2))],
                             [(math.pi)/(6*math.sqrt(2))],
                             [0]])
hat_theta = torch.tensor([[0,0,(math.pi)/(6*math.sqrt(2))],
                          [0,0,(math.pi)/(-6*math.sqrt(2))],
                          [-(math.pi)/(6*math.sqrt(2)),(math.pi)/(6*math.sqrt(2)),0]])
norm_theta = torch.norm(vector_theta)

I = torch.eye(3)
rotation_b = I + ((math.sin(norm_theta))/(norm_theta))*hat_theta+((1-math.cos(norm_theta))/(norm_theta**2))*(hat_theta@hat_theta)

print(rotation_b)
print(norm_theta)
x_c = torch.tensor([[1.],[1.],[1.]])
t_c = torch.tensor([[0.],[1.],[2.]])

x_b = rotation_b @ x_c + t_c
print(x_b)
phi = math.pi/6
theta = math.pi/3
psi = math.pi/4

r_x = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, math.cos(phi), -math.sin(phi)],
    [0.0, math.sin(phi), math.cos(phi)]
])

r_y = torch.tensor([
    [math.cos(theta), 0.0, math.sin(theta)],
    [0.0, 1.0, 0.0],
    [-math.sin(theta), 0.0, math.cos(theta)]
])

r_z = torch.tensor([
    [math.cos(psi), -math.sin(psi), 0.0],
    [math.sin(psi), math.cos(psi), 0.0],
    [0.0, 0.0, 1.0]
])

rotation_a = r_z @ r_y @ r_x
print(rotation_a)
phi = math.pi/6
theta = math.pi/3
psi = math.pi/4

r_x = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, math.cos(phi), -math.sin(phi)],
    [0.0, math.sin(phi), math.cos(phi)]
])

r_y = torch.tensor([
    [math.cos(theta), 0.0, math.sin(theta)],
    [0.0, 1.0, 0.0],
    [-math.sin(theta), 0.0, math.cos(theta)]
])

r_z = torch.tensor([
    [math.cos(psi), -math.sin(psi), 0.0],
    [math.sin(psi), math.cos(psi), 0.0],
    [0.0, 0.0, 1.0]
])

rotation_a = r_z @ r_y @ r_x
print(rotation_a)

t_b = torch.tensor([[1],[1],[0]])
x_a = rotation_a @ x_b + t_b
print(x_a)
```