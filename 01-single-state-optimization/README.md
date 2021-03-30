## Single-state Optimization

To use the techniques of the single-state methods to minimize the objective function in search space.

### Objective function

![image](objective-function.png)

### Single-state optimization methods

* Exhaustive Enumeration

* Random methods
  1. Random jump
  2. Random walk
  
* Gradient-based methods
  1. Steepest Gradient Descent (SGD) method
  2. Newton’s method
  3. Marquardt’s method
  
* Stochastic Local Search methods
  1. Iterated Local Search (ILS) method
  2. Simulated Annealing (SA) method
  3. Tabu Search (TS) method
  
### Implementation on the optimization algorithms

#### Requirement

* python 3
  * numpy
  * itertools

#### Implementation

#### 1. Description

```
01-single-state-optimization/
├── README.md
├── objective-function.png
├── optimization.py
└── report.pdf
```
In [optimization.py](optimization.py), single-state optimization methods are degined as functions under the object, *single_state_optimizer()*. In addition, the objective function *f* is also defined in it.

#### 2. Initialization

* Search space

```python
x_min = np.array([-6, -6]).reshape([-1, 1])
x_max = np.array([6, 6]).reshape([-1, 1])
```

* Coefficients of the objective function

```python
a = np.array([3, 14]).reshape([-1, 1])
c = np.array([6, 7]).reshape([-1, 1])
b = -17
d = -19
```

* Initialize the optimizer

```python
optimizer = single_state_optimizer(a, b, c, d, x_min, x_max, decimal = 4) 
```

#### 3. Single-state ptimization methods

Input settings refer to the [report.pdf](report.pdf) or the [optimization.py](optimization.py) files.

###### 1. Exhaustive Enumeration

```python
opt_pts, opt_sol, num_iter = optimizer.exhaustive_enum(x_grid_interval)
```

###### 2. Random Methods
###### 2-1. Random Jump

```python
opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_jump(expected_opt_pts, tolerate_error, random_seed, random_interval, grid_size_scale, jump_times_per_grid)
```

###### 2-2. Random Walk

###### 2-2-1. Decaying lambda

```python
opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_walk1(x0, N, RN, random_seed, random_interval, lam, eps)
```
###### 2-2-1. Optimal lambda

```python
opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_walk2(x0, N, RN, random_seed, random_interval, ITER)
```

###### 3. Gradient-based Methods
###### 3-1. Steepest Descent Method (SGD)

```python
opt_pts, opt_sol, num_iter = optimizer.SGD(x0, eps)
```

###### 3-2. Newton's Method

```python
opt_pts, opt_sol, num_iter = optimizer.newtons(x0, eps)
```

###### 3-3. Marquardt's Method

```python
opt_pts, opt_sol, num_iter = optimizer.marquardts(x0, eps, alpha, c1, c2)
```

###### 4. Stochastic Local Search Methods
###### 4-1. Iterated Local Search (ILS) method 

```python
opt_pts, opt_sol, num_iter = optimizer.ILS(x0, random_seed, random_interval, alpha, scale, ratio, num_search, ITER)
```

###### 4-2. Simulated Annealing (SA) method

```python
opt_pts, opt_sol, num_iter = optimizer.SA(x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER)
```

###### 4-3. Tabu Search (TS) method

```python
opt_pts, opt_sol, num_iter = optimizer.TS(x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER, maxTabuSize, num_neighbors)
```

### Results

The details of the optimization results (e.g. optimal point, optimum and number of iterations, etc.) and the comparison of the methods are illustrated in [report.pdf](report.pdf).
