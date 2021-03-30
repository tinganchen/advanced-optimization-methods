import numpy as np
import matplotlib.pyplot as plt
import itertools

import warnings
warnings.simplefilter("ignore")

'''
## Description

use techniques of single-state methods to
find the minimum point of a simple function
'''
###

'''
## Optimizers

# 1. Exhaustive Enumeration: exhaustive_enum
# 2. Random methods: random_jump & random_walk1 & random_walk2
# 3. Gradient-based Methods: SGD & newtons & marquardts
# 4. Stochastic Local Search Methods: ILS & SA & TS
'''
class single_state_optimizer():
    def __init__(self, a, b, c, d, x_min, x_max, decimal):
        super(single_state_optimizer, self).__init__()
        
        self.x_min = x_min
        self.x_max = x_max
        
        self.decimal = decimal
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    # objective function
    def f(self, x):
        return float((np.matmul(np.transpose(self.a), x) + self.b)**2 + (np.matmul(np.transpose(self.c), x) + self.d)**2)
  
    ## 1. Exhaustive Enumeration (Grid search)
    def exhaustive_enum(self, x_grid_interval):
        x_min = self.x_min
        x_max = self.x_max
        
        x_num_segment = (x_max - x_min) / x_grid_interval
        x_grids = [np.linspace(x_min[i], x_max[i], int(x_num_segment[i])) for i in range(x_num_segment.shape[0])]
        
        search_space = list(itertools.product(*x_grids))
        
        minimum = np.inf
        
        COUNT_ITER = 0
        for x in search_space:
            sol = self.f(np.array(x).reshape([-1, 1]))
            if sol < minimum:
                minimum = sol
                opt_x = np.array(x).reshape([-1, 1])
                COUNT_ITER += 1
                
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER
    
    
    ## 2. Random Methods
    ## 2-1. Random Jump
    def random_jump(self, expected_opt_pts, tolerate_error, random_seed, random_interval, grid_size_scale, jump_times_per_grid):
        x_min = self.x_min
        x_max = self.x_max
        
        expected_opt_pts = np.array(expected_opt_pts).reshape([-1, 1])
        
        opt_x = np.repeat(np.inf, len(expected_opt_pts)).reshape([-1, 1])
        
        minimum = np.inf
        
        COUNT_ITER = 0
        COUNT_SEED = 0
        while self.euclidian_dist(expected_opt_pts, opt_x) >= tolerate_error: # stop_criterion
            x_num_segment = int(1 / grid_size_scale)
            x_grids = [np.linspace(x_min[i], x_max[i], x_num_segment) for i in range(x_min.shape[0])]
            
            x_ranges = []
            for xi_grids in x_grids:
                x_centers = [(xi_grids[i] + xi_grids[i+1]) / 2 for i in range(len(xi_grids)-1)]
                radius = x_centers[0] - xi_grids[0]
                x_lbs = np.array(x_centers) - radius
                x_ubs = np.array(x_centers) + radius
                x_ranges.append(list(zip(x_lbs, x_ubs)))
                
            
            search_space = list(itertools.product(*x_ranges))
            
         
            for x_bounds in search_space:
                for _ in range(jump_times_per_grid):
                    x = []
                    for (x_lb, x_ub) in x_bounds:
                        np.random.seed(random_seed)
                        xi = np.random.uniform(x_lb, x_ub) 
                        x.append(xi)
                        random_seed += random_interval
                        COUNT_SEED += 1
                        
                    x = np.array(x).reshape([-1, 1])
                    sol = self.f(x)
                    
                    if sol < minimum:
                        minimum = sol
                        opt_x = x
                        opt_grid = x_bounds
          
                    
                    COUNT_ITER += 1
            #print(COUNT_SEED)
            x_min = np.array([i[0] for i in opt_grid]).reshape([-1, 1])
            x_max = np.array([i[1] for i in opt_grid]).reshape([-1, 1])
            
                
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER, COUNT_SEED
    
    def euclidian_dist(self, expected_opt_pts, opt_x):
        return (sum((expected_opt_pts - opt_x)**2))**0.5
        
    ## 2-2. Random Walk
    ## 2-2-1. lam /= 2
    def random_walk1(self, x0, N, RN, random_seed, random_interval, lam, eps):
        
        opt_x = x0.reshape([-1, 1])
        minimum = self.f(x0)
        
        COUNT_ITER = 0
        COUNT_SEED = 0
        while (lam > eps): # or (self.euclidian_dist(expected_opt_pts, opt_x) >= tolerate_error):
            for i in range(N):
                r_js = []
                for j in range(RN):
                    np.random.seed(random_seed)
                    r_j = np.random.uniform(-1, 1, 2).reshape([-1, 1])
                    r_js.append(r_j)
                    R = sum(np.sqrt([sum(r**2) for r in r_js]))
                    
                    if R >= 1:
                        r_js.pop()
                    
                for r_j in r_js:  
                    u_j = r_j / np.sqrt(sum(r_j**2))
                    x = opt_x + lam * u_j
                    if self.f(x) < minimum:
                        opt_x = x
                        minimum = self.f(x)
                    
                    COUNT_ITER += 1
                    COUNT_SEED += 1
                    
            lam /= 2
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER, COUNT_SEED

    ## 2-2-2. lam*
    def random_walk2(self, x0, N, RN, random_seed, random_interval, ITER):
        
        opt_x = x0.reshape([-1, 1])
        minimum = self.f(x0)
        
        COUNT_ITER = 0
        COUNT_SEED = 0
        while (COUNT_ITER < ITER): # or (self.euclidian_dist(expected_opt_pts, opt_x) >= tolerate_error):
            for i in range(N):
                r_js = []
                for j in range(RN):
                    np.random.seed(random_seed)
                    r_j = np.random.uniform(-1, 1, 2).reshape([-1, 1])
                    r_js.append(r_j)
                    R = sum(np.sqrt([sum(r**2) for r in r_js]))
                    
                    if R >= 1:
                        r_js.pop()
                    
                for r_j in r_js:  
                    u_j = r_j
                    lam = self.lam_star_random_walk(opt_x, u_j)
                    x = opt_x + lam * u_j
                    if self.f(x) < minimum:
                        opt_x = x
                        minimum = self.f(x)
                    
                    COUNT_ITER += 1
                    COUNT_SEED += 1

        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER, COUNT_SEED
    
    def lam_star_random_walk(self, x, u): 
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        K = -(np.matmul(np.transpose(a), x) + b) * np.matmul(np.transpose(a), u)
        K -= (np.matmul(np.transpose(c), x) + d) * np.matmul(np.transpose(c), u)
        return K / ((np.matmul(np.transpose(a), u)**2 + np.matmul(np.transpose(c), u)**2))
    
    
    ## 3. Gradient-based Methods
    ## 3-1. Steepest Descent Method (SGD)
    def SGD(self, x0, eps):
        opt_x = x0.reshape([-1, 1])
        opt_grad = self.gradient(opt_x)
        
        COUNT_ITER = 0
        while (np.mean(np.abs(opt_grad)) > eps):
            opt_grad = self.gradient(opt_x)
            opt_x = opt_x - self.lam_star_SGD(opt_x, opt_grad) * opt_grad
            COUNT_ITER += 1
            
        minimum = self.f(opt_x)
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER

    def gradient(self, x):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        grad = 2*(np.matmul(np.transpose(a), x) + b) * a 
        grad += 2*(np.matmul(np.transpose(c), x) + d) * c
        return grad
    
    def lam_star_SGD(self, x, grad): 
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        K = (np.matmul(np.transpose(a), x) + b) * np.matmul(np.transpose(a), grad)
        K += (np.matmul(np.transpose(c), x) + d) * np.matmul(np.transpose(c), grad)
        return K / ((np.matmul(np.transpose(a), grad)**2 + np.matmul(np.transpose(c), grad)**2))
    
    ## 3-2. Newton's Method
    def newtons(self, x0, eps):
        opt_x = x0.reshape([-1, 1])
        opt_grad = self.gradient(opt_x)
        
        COUNT_ITER = 0
        while (np.mean(np.abs(opt_grad)) > eps):
            opt_grad = self.gradient(opt_x)
            J = self.jacobian()
            opt_x = opt_x - np.matmul(np.linalg.inv(J), opt_grad)
            COUNT_ITER += 1
            
        minimum = self.f(opt_x)
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER

    def jacobian(self): 
        a1, a2 = self.a
        c1, c2 = self.c
        J = np.zeros([2, 2])
        J[0, 0] = 2 * (a1**2 + c1**2)
        J[1, 1] = 2 * (a2**2 + c2**2)
        J[0, 1] = 2 * (a1*a2 + c1*c2)
        J[1, 0] = J[0, 1]
        return J
    
    ## 3-3. Marquardt's Method
    def marquardts(self, x0, eps, alpha, c1, c2):
        opt_x = x0.reshape([-1, 1])
        opt_grad = self.gradient(opt_x)
        
        COUNT_ITER = 0
        while (np.mean(np.abs(opt_grad)) > eps):
            opt_grad = self.gradient(opt_x)
            J = self.jacobian()
            opt_x_new = opt_x - np.matmul(np.linalg.inv(J + alpha * np.eye(J.shape[0])), opt_grad)
            
            if self.f(opt_x_new) < self.f(opt_x):
                alpha *= c1
            else:
                alpha *= c2
            
            opt_x = opt_x_new
            
            COUNT_ITER += 1
            
        minimum = self.f(opt_x)
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER
    
    
    ## 4. Stochastic Local Search Methods
    ## 4-1. Iterated Local Search Method (ILS)
    def ILS(self, x0, random_seed, random_interval, alpha, scale, ratio, num_search, ITER):
        opt_x_new = x0.reshape([-1, 1])
        opt_x = x0.reshape([-1, 1])
        
        minimum = np.inf
       
        COUNT_ITER = 0
        while COUNT_ITER < ITER:
            opt_x = opt_x_new
            for _ in range(num_search):
                np.random.seed(random_seed)
                r = np.random.uniform(0, alpha)
                random_seed += random_interval
                
                np.random.seed(random_seed)
                theta = np.random.uniform(0, 2*np.pi)
                random_seed += random_interval
                
                opt_x_tmp = opt_x + r * np.array([np.cos(theta), np.sin(theta)]).reshape([-1, 1])
                
                COUNT_ITER += 1
                
                if self.f(opt_x_tmp) < minimum:
                    opt_x_new = opt_x_tmp
                    minimum = self.f(opt_x_new)
                    opt_r = r
                    
            if opt_r / alpha > ratio:
                alpha *= scale
            
            
        opt_x = opt_x_new    
            
        minimum = self.f(opt_x)
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER
    
    ## 4-2. Stimulated Annealing Method (SA)
    def SA(self, x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER):
        opt_x_new = x0.reshape([-1, 1])
        opt_x = x0.reshape([-1, 1])
        
        minimum = np.inf
        
        x_min = self.x_min
        x_max = self.x_max
        
        np.random.seed(random_seed)
        x_seeds = [np.random.uniform(x_min[i], x_max[i], num_init_seeds) for i in range(len(x_min))]
        random_seed += random_interval
        
        x_seeds = list(zip(*x_seeds))
        
        T = 0
        for x_seed in x_seeds:
            fi = self.f(np.array(x_seed).reshape([-1, 1]))
            T += fi
        
        T /= num_init_seeds
        
        COUNT_ITER = 0
        while COUNT_ITER < ITER:
            np.random.seed(random_seed)
            r = np.random.uniform(0, alpha)
            random_seed += random_interval
            
            np.random.seed(random_seed)
            theta = np.random.uniform(0, 2*np.pi)
            random_seed += random_interval
            
            opt_x_new = opt_x + r * np.array([np.cos(theta), np.sin(theta)]).reshape([-1, 1])
            
            if self.f(opt_x_new) < minimum:
                minimum = self.f(opt_x_new)
                opt_r = r
            
            diff = self.f(opt_x_new) - self.f(opt_x)
            
            metropolis = np.exp(-diff / (temp_reduce**COUNT_ITER * T))
            
            np.random.seed(random_seed)
            rand = np.random.uniform(0, 1)
            random_seed += random_interval

            if diff < 0 or rand < metropolis:
                opt_x = opt_x_new
            
            if opt_r / alpha > ratio:
                alpha *= scale
            
            COUNT_ITER += 1
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER
    
    ## 4-3. Tabu Search (TS) Method
    def TS(self, x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER, maxTabuSize, num_neighbors):
        opt_x_new = x0.reshape([-1, 1])
        opt_x = x0.reshape([-1, 1])
        
        minimum = np.inf
        
        x_min = self.x_min
        x_max = self.x_max
        
        np.random.seed(random_seed)
        x_seeds = [np.random.uniform(x_min[i], x_max[i], num_init_seeds) for i in range(len(x_min))]
        random_seed += random_interval
        
        x_seeds = list(zip(*x_seeds))
        
        T = 0
        for x_seed in x_seeds:
            fi = self.f(np.array(x_seed).reshape([-1, 1]))
            T += fi
        
        T /= num_init_seeds
        
        # Tabu
        tabuList = []
        tabuList.append(opt_x)
        
        
        COUNT_ITER = 0
        while COUNT_ITER < ITER:
            for _ in range(num_neighbors):
                np.random.seed(random_seed)
                r = np.random.uniform(0, alpha)
                random_seed += random_interval
                
                np.random.seed(random_seed)
                theta = np.random.uniform(0, 2*np.pi)
                random_seed += random_interval
                
                opt_x_tmp = opt_x + r * np.array([np.cos(theta), np.sin(theta)]).reshape([-1, 1])
                
                # check if it is in tabuList
                inTabuList = max([np.mean(opt_x_tmp == tabu) == 1 for tabu in tabuList])
                
                if (not inTabuList) and (self.f(opt_x_tmp) < self.f(opt_x_new)):
                    opt_x_new = opt_x_tmp
            
            if self.f(opt_x_new) < minimum:
                minimum = self.f(opt_x_new)
                opt_r = r
            
            diff = self.f(opt_x_new) - self.f(opt_x)
            
            metropolis = np.exp(-diff / (temp_reduce**COUNT_ITER * T))
            
            np.random.seed(random_seed)
            rand = np.random.uniform(0, 1)
            random_seed += random_interval

            if diff < 0 or rand < metropolis:
                opt_x = opt_x_new
            
            if opt_r / alpha > ratio:
                alpha *= scale
            
            tabuList.append(opt_x_new)
            
            if (len(tabuList) > maxTabuSize):
                tabuList.pop(0)
            
            COUNT_ITER += 1
        
        return [i[0] for i in np.round(opt_x, self.decimal)], np.round(minimum, self.decimal+2), COUNT_ITER
    
    
    
''' search space ''' 
x_min = np.array([-6, -6]).reshape([-1, 1])
x_max = np.array([6, 6]).reshape([-1, 1])


''' initialize the optimizer ''' 
a = np.array([3, 14]).reshape([-1, 1])
c = np.array([6, 7]).reshape([-1, 1])

b = -17
d = -19

optimizer = single_state_optimizer(a, b, c, d, x_min, x_max, 4) 


''' find the optimal point and solution '''
## 1. Exhaustive Enumeration
x_grid_interval = np.array([0.01, 0.01]).reshape([-1, 1])

opt_pts, opt_sol, num_iter = optimizer.exhaustive_enum(x_grid_interval)
### ([2.3269, 0.7156], 0.000853, 7215)


## 2. Random Methods
## 2-1. Random Jump
expected_opt_pts = [2.33, 0.72] # set to the optimal solution of Exhaustive Enumeration
tolerate_error = 0.05
random_seed = 1
random_interval = 20

grid_size_scale = 1/3
jump_times_per_grid = 20

opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_jump(expected_opt_pts, tolerate_error, random_seed, random_interval, grid_size_scale, jump_times_per_grid)
### ([2.3433, 0.7207], 0.025215, 320, 640)

## 2-2. Random Walk 
x0 = np.zeros([2, 1])

N = 100
RN = 5 # num_samples

random_seed = 0
random_interval = 1

lam = 1
eps = 0.05

## 2-2-1. lam = lam / 2
opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_walk1(x0, N, RN, random_seed, random_interval, lam, eps)
### ([0.3318, 1.4628], 65.849032, 1000, 1000)

## 2-2-2. lam*
ITER = 1000

opt_pts, opt_sol, num_iter, num_rand_seed = optimizer.random_walk2(x0, N, RN, random_seed, random_interval, ITER)
### ([0.3246, 1.431], 65.558849, 1000, 1000)


## 3. Gradient-based Methods
x0 = np.zeros([2, 1])

eps = 0.01

## 3-1. Steepest Descent Method (SGD)
opt_pts, opt_sol, num_iter = optimizer.SGD(x0, eps)
### ([2.3333, 0.7143], 0.0, 10)

## 3-2. Newton's Method
opt_pts, opt_sol, num_iter = optimizer.newtons(x0, eps)
### ([2.3333, 0.7143], 0.0, 2)

## 3-3. Marquardt's Method
alpha = 10**4
c1 = 0.25
c2 = 2

opt_pts, opt_sol, num_iter = optimizer.marquardts(x0, eps, alpha, c1, c2)
### ([2.3333, 0.7143], 0.0, 9)


## 4. Stochastic Local Search Methods
x0 = np.zeros([2, 1])

random_seed = 1
random_interval = 20

alpha = 1
scale = 0.999
ratio = 0.8

## 4-1. Iterated Local Search (ILS) method 
num_search = 20
ITER = 500

opt_pts, opt_sol, num_iter = optimizer.ILS(x0, random_seed, random_interval, alpha, scale, ratio, num_search, ITER)
### ([2.3422, 0.7095], 0.002006, 500)

## 4-2. Simulated Annealing (SA) method
num_init_seeds = 5
temp_reduce = 0.5
ITER = 500

opt_pts, opt_sol, num_iter = optimizer.SA(x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER)
### ([2.3374, 0.7068], 0.009437, 500)

## 4-3. Tabu Search (TS) method
num_init_seeds = 5
temp_reduce = 0.5

ITER = 500

maxTabuSize = 7
num_neighbors = 20

opt_pts, opt_sol, num_iter = optimizer.TS(x0, random_seed, random_interval, alpha, scale, ratio, num_init_seeds, temp_reduce, ITER, maxTabuSize, num_neighbors)
### ([2.3335, 0.7141], 2e-06, 500)

