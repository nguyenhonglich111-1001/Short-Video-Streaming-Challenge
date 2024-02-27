import numpy as np
import run
import importlib
def write_arr_to_file(arr):
    with open('./constant/reward_hyper_params.txt', 'w') as f:
        for e in arr:
            f.write(str(e)+'\n')

def read_params():
    with open('./constant/reward_hyper_params.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = float(lines[i].split('\n')[0])

        print(lines)
        # get fist 4 line trainable
        return lines[:4]
    

def sign_array(a):
    b = np.zeros_like(a)
    b[a > 0] = 1
    b[a < 0] = -1
    return b
# Numerical gradient approximation
def numerical_gradient(f, params, epsilons):
    gradient = np.zeros_like(params)
    for i, param in enumerate(params):
        if learning_rates[i] == 0:
            continue
        params_plus = params.copy()
        params_plus[i] += epsilons[i]
        params_minus = params.copy()
        params_minus[i] -= epsilons[i]

        write_arr_to_file(params_plus)
        gradient_plus = f()
        write_arr_to_file(params_minus)
        gradient_minus = f()
        print('calc', i, [params_plus, params_minus], [
              gradient_plus, gradient_minus], 2 * epsilons[i])
        gradient[i] = (gradient_plus - gradient_minus) / (2 * epsilons[i])
    return gradient

# Gradient descent optimization
def gradient_descent(f, initial_params, learning_rates, num_iterations, epsilons):
    params = initial_params.copy()
    for _ in range(num_iterations):
        gradient = numerical_gradient(f, params, epsilons=epsilons)

        gradient = sign_array(gradient)
        params += learning_rates * gradient * epsilon_params
        print('adjust volume', learning_rates * gradient * epsilon_params)
    write_arr_to_file(params)
    return params

def test_f(x,y):
    return 20*x-x*x+30*y-y*y+x*y
 
def run_func():
    importlib.reload(run)
    f = run.test_user_samples

    return f(*run_params)

def test_consistent_result():
    res = []
    for i in range(4):
        res.append(run_func())

    print("consistent result", res)


run_params = [False, True, run.args.quickstart,
              run.args.trace, run.SAMPLE_COUNT]
initial_params = np.array(read_params())
learning_rates = np.array([1,1,1,1])
epsilon_params = np.array([1e-4,1e-4,1e-4,1e-5])

print('Start',initial_params*learning_rates)
if __name__ == '__main__':

    f = run_func
    optimized_params = gradient_descent(
        f, initial_params, learning_rates=learning_rates, num_iterations=10, epsilons=epsilon_params)
    max_value = f()
    print("Maximum value of the objective function:", max_value)
    

