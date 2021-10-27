from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt

data = load_protein(info=1)

X, y = data[:,:-1], data[:, -1:]

X = max_min_scale(X)

n_train = 3000
train_x, test_x, train_y, test_y = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

#initiate hyperparameters
dims = len(X[0])
lengthscales = np.random.uniform(low=0.01,high=1.0,size=dims).astype('float64')
sigma = 10.

kernel = RBFKernel(ard_num_dims = dims)
kernel.initialize(lengthscale=lengthscales)

#form covariance matrix
K_XX = kernel(torch.Tensor(train_x), torch.Tensor(train_x)).evaluate().detach().numpy()

#form likelihood matrix
K_hat = K_XX + sigma ** 2 * np.eye(n_train)

#compute exact solution
x_sol = np.linalg.inv(K_hat) @ train_y

#cg run
cg_run = lin_cg(K_hat, train_y)
error = cg_run[0] - x_sol
error = np.linalg.norm(error)

print("\nCG iterations: ",cg_run[2])
print("\nError: ",error)