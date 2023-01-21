# How do Synsin handle with dataset
1. What data can be used,
- [RealEstate10K](https://google.github.io/realestate10k/download.html)

2. What is the input format of the data?
- images and cameras

# What is the workflow of the Synsin
1. Input an image $I$, and change in pose $T$
2. Generate spatial features $F$, regress a depth map $D$
3. $F$ are projected to point cloud features $P$.
4. $P$ are transofrmed according to $T$ and rendered to $\bar{F}$ and refined to the final image $I_G$

# How to represent the change of pose $T$
1. Rotation: $\theta$, $\phi$
2. Translation: tx, ty, tz

# Why the naive point cloud renderer is non-differentiable
- A naive renderer: 
    - $p_i$ -> one pixel or a small region (_footprint_)
    - $p_i$ sorted in depth using a z-buffer
    - in the new view, nearest point in depth is chosen
- Why is it non-differentiable:
    - my understanding: the new point is rendered with the nearest depth old point, but if the depth for old point change, the position of nearest depth will change accordingly. So for a renderer, when the output change, it doesn't mean the depth value of input changed, but both the value and the position of inpot depth changed.
    - explanation in the paper: 
        1. small neighborhoods: in rendered view, each new point only has a few gradients correspoonding to old points. 
        2. hard z-buffer: the same as my understanding

# What are the solutions from the Synsin
1. A point $p_i$ is splatted to a region with center $p_{i_c}$, radius $r$
2. Accumulated with K-nearest neighbors.

# Data format in RealEstate10K

Line 1. timestamp

Line 2-6. camera intrinsics (float: focal_length_x, focal_length_y, principal_point_x, principal_point_y)

Line 7-19. camera pose (3x4 matrix in row major order)