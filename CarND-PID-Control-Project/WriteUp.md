# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

### How to run the simulation
1. Navigate to `CarND-PID-Control-Project` folder 
2. run the commands below in the terminal
```
mkdir build && cd build
cmake .. && make
./pid
```
3. Click on the "Simulator" button in the bottom of the Udacity workspace.
4. Double-click the "Simulator" icon in that desktop to start the simulator.
## Introduce of the PID controller

PID control is a technique to correct vehicle steering Angle, throttle threshold and other behaviors according to loss components of P(proportion), I(integral) and D(differential) given vehicle crossing trajectory error.

In the Udacity automotive simulator, the CTE value is read from the data message sent by the simulator, and the PID controller updates the error value and predicts the steering angle based on the total error.
![](https://raw.githubusercontent.com/aaron7yi/CarND-PID-Control-Project/master/twiddle.png)
**The P (Proportional)  gain**

The proportional term computes an output proportional to the cross-track error. A pure P - controller is unstable and at best oscillates about the setpoint. The proportional gain contributes a control output to the steering angle of the form `-K_p cte`with a positive constant `K_p`.

**The D (Differential) gain**

The oscillations caused by purely D control can be mitigated by a term proportional to the derivative of the cross-track error. The derivative gain contributes a control output of the form `-K_d d/dt cte`, with a positive constant `K_d`.

**The I (Integral) gain**

A third contribution is given by the integral gain which simply sums up the cross-track error over time. The corresponding contribution to the steering angle is given by `-K_i sum(cte)`. 

### Discussion
How do I tune the P, I, D value:
**P value:** 
Adding the ratio Kp coefficient can reduce the deviation quickly. The smaller the
Kp coefficient is, the weaker the control effect is and the slower the system
response is. On the contrary, the larger the Kp coefficient is, the stronger the control
effect is and the faster the system response is. However, if the Kp is too large, the
system will produce large overshooting and oscillation, which will lead to poor
stability of the system. Therefore, Kp should be selected carefully so that it will not
produce large oscillation while reducing the deviation. However, the static error can
not be completely eliminated simply by adding Kp coefficient.
**D value:** 
By adding differential Kd coefficient, the system can respond in time when there
is a deviation at the beginning, and can generate control according to the variation
trend of the deviation, which can reduce the overshoot of the system and overcome
the oscillation. Too large Kd coefficient is easy to cause system instability.
**I value:**
Adding the integral ki coefficient can eliminate the static error and improve the
error free degree of the system. The strength of the integral action depends on the
Ki coefficient. The larger the Ki coefficient is, the slower the integral speed is, and
the weaker the integral action is. On the contrary, the smaller the Ki coefficient is,
the faster the integral speed is, and the stronger the integral action is. Too strong
integration will increase the overshoot of the system and even make the system
oscillate. Moreover, the introduction of integral coefficient will reduce the response
speed of the system.
### Simulation result
[![Test video](https://viewsjgdpiwcxq.cn1-udacity-student-workspaces.com/files/home/workspace/CarND-PID-Control-Project/video/1.mp4)]
[![Test video](https://viewsjgdpiwcxq.cn1-udacity-student-workspaces.com/files/home/workspace/CarND-PID-Control-Project/video/2.mp4)]
[![Test video](https://viewsjgdpiwcxq.cn1-udacity-student-workspaces.com/files/home/workspace/CarND-PID-Control-Project/video/3.mp4)]
### Future Improvements
- Adjust throttle to be controlled by the cross track error so that the vehicle can run faster
- Try to automatically tune gains by using twiddle algorithm