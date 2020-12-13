import numpy as np
import matplotlib.pyplot as plt


# Constant parameters
storage_cap = 40
purchase_cap = 40


def generate_random_data(T, seed=4208):
    # Set seed
    np.random.seed(seed)
    # Draw from uniform distribution and round floats to 1 decimal point
    purchase_cost = np.around(np.random.uniform(10, 20, size=T), 1).tolist()
    holding_cost = np.around(np.random.uniform(5, 15, size=T), 1).tolist()
    # Draw from uniform integer distribution
    # max is exclusive for randint, so using 21 instead of 20
    demand = np.random.randint(0, 21, size=T).tolist()
    return(purchase_cost, holding_cost, demand)


def recursion_function(f_future, demand, purchase_cost,
                       holding_cost):
    # Extract parameters for period
    d_t = demand.pop(-1)
    c_t = purchase_cost.pop(-1)
    h_t = holding_cost.pop(-1)

    if len(demand) > 0:
        # Not the deepest level
        x = [None] * (storage_cap + 1)
        f = [None] * (storage_cap + 1)
        # Iterate through all possible starting inventories
        # The index of these arrays corresponds to the starting inventory
        for i in range(storage_cap + 1):
            x[i], f[i] = optimizer_function(d_t, c_t, h_t, i, f_future)
        # Go one level deeper
        x_arr, f_arr, inv_arr = recursion_function(f, demand,
                                                   purchase_cost,
                                                   holding_cost)
        # Work backwards up recursion stack to get decision values
        start_inv = inv_arr[-1]
        x_arr.append(x[start_inv])
        f_arr.append(x[start_inv])
        inv_arr.append(start_inv + x[start_inv] - d_t)
    else:
        # At deepest level, find minimum future cost
        x, f = optimizer_function(d_t, c_t, h_t, 0, f_future)
        end_inv = x - d_t

        x_arr = [x]
        f_arr = [f]
        inv_arr = [end_inv]
    return(x_arr, f_arr, inv_arr)


def optimizer_function(d_t, c_t, h_t, start_inv, f_future):
    # Set upper and lower x bounds based on constraints
    lb = int(np.max([0, d_t - start_inv]))
    ub = int(np.min([purchase_cap, d_t + storage_cap - start_inv]))
    if ub < lb:
        # Infeasible, set a big number
        x = 0
        f = 10 ** 10
    else:
        x_arr = list(range(lb, ub + 1))
        f_arr = []
        # Find global minimum for stage by iterating through all possible x
        for x in x_arr:
            final_inv = start_inv + x - d_t
            future_cost = f_future[final_inv]
            f_arr.append(c_t * x + h_t * (start_inv + x - d_t) + future_cost)
        f = min(f_arr)
        x = x_arr[f_arr.index(f)]
    return(x, f)


def main():
    # Generate some data
    purchase_cost, holding_cost, demand = generate_random_data(T=10)
    # Copy the demand list since it will be modified
    demand_for_plot = demand.copy()
    # For last period, future cost is 0
    f_future = [0] * (storage_cap + 1)
    # Run recursion function
    x_arr, fc_arr, inv_arr = recursion_function(f_future, demand,
                                                purchase_cost,
                                                holding_cost)
    # Plot data
    plt.figure()
    plt.plot(demand_for_plot, label='demand')
    plt.plot(inv_arr, label='inventory')
    plt.plot(x_arr, label='purchased')
    plt.legend()
    plt.show()
    return(x_arr, fc_arr)


if __name__ == "__main__":
    main()
