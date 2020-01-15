# SIMPLE PENDULUM

**STANDALONE TOPOLOGY**

------------------------------------------------

### **Summary**
A pendulum is simply a weight suspended from a pivot so that it can swing freely. [wikipedia](https://en.wikipedia.org/wiki/Pendulum).

--------------------------------------

### **Topology Layout**
The mechanism consists of 1 Body + 1 Ground. Therefore, total system coordinates -including the ground- is $n=n_b\times7 = 2\times7 = 14$, where $n_b$ is the total number of bodies.[^1]

The list of bodies is given below:
- Pendulum  $l_1$.

The system connectivity is as follows:

<center>

| Joint Name  | Body i         | Body j         | Joint Type | $n_c$ |
|:-----------:|:-------------- |:-------------- | ---------- | ----- |
| a           | Ground         | Pendulum 1     | Revolute   | 5     |

</center>

<br/>

Hence, the total number of constraints equations is:
$$ n_{c} = n_{c_j} + n_{c_p} + n_{c_g} = 5 + (2\times 1) + 6 = 13 $$

where:
* $n_{c_j}$ is the joints constraints.
* $n_{c_p}$ is the euler-parameters normalization constraints.
* $n_{c_g}$ is the ground constraints.

Therefore, the resulting **DOF** is $ n - n_c = 14 - 13 = 1 $

------------------------------------------------------
<br/>

[^1]: The tool uses [euler-parameters](https://en.wikibooks.org/wiki/Multibody_Mechanics/Euler_Parameters) -which is a 4D unit quaternion- to represents bodies orientation in space. This makes the generalized coordinates used to fully define a body in space to be **7,** instead of **6**, it also adds an algebraic equation to the constraints that ensures the unity/normalization of the body quaternion. This is an important remark as the calculations of the degrees-of-freedom depends on it.

