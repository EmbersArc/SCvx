# SuccessiveConvexification
Implementation of [Successive Convexification: A Superlinearly Convergent Algorithm for Non-convex Optimal Control Problems
](https://arxiv.org/abs/1804.06539) by Yuanqi Mao, Michael Szmuk and Behcet Acikmese

Reqires `matplotlib`, `numpy`, `scipy` and `cvxpy`.


This framework provides an easy way to implement your own models.

Fixed- and free-final-time optimization is possible.

The following models are currently implemented:

- Rocket trajectory model with free-final-time based on
[Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time](https://arxiv.org/abs/1802.03827)
by Michael Szmuk and Behçet Açıkmeşe.

![](https://i.imgur.com/W6E0rgL.png)

[Video example of generated trajectories](https://gfycat.com/InbornCoarseArcticseal)

- Differential drive robot path planning with free-final-time and non-convex obstacle constraints:
![](https://i.imgur.com/xNaD5eP.png)

- 2D rocket landing problem
[Feed-forward input tested in a box2d physics simulation](https://gfycat.com/DaringPortlyBlacklab)
