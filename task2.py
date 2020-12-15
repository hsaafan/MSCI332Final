import numpy as np
import matplotlib.pyplot as plt
import time


# Constant parameters
storage_cap = 40
purchase_cap = 40

#Set Seed
seed = 4208
np.random.seed(seed)

def generate_random_data(T):
    # Draw from uniform distribution and round floats to 1 decimal point
    purchase_cost = np.around(np.random.uniform(10, 20, size=T), 1).tolist()
    holding_cost = np.around(np.random.uniform(5, 15, size=T), 1).tolist()
    shortage_cost = np.around(np.random.uniform(15, 35, size=T), 1).tolist()
    # Draw from uniform integer distribution
    # max is exclusive for randint, so using 21 instead of 20
    demand = np.random.randint(0, 21, size=T).tolist()
    return(purchase_cost, holding_cost, demand, shortage_cost)


def recursion_function(f_future, demand, purchase_cost,
                       holding_cost, shortage_cost):
    # Extract parameters for period
    d_t = demand.pop(-1)
    c_t = purchase_cost.pop(-1)
    h_t = holding_cost.pop(-1)
    s_t = shortage_cost.pop(-1)

    if len(demand) > 0:
        # Not the deepest level
        x = [None] * (storage_cap + 1)
        f = [None] * (storage_cap + 1)
        # Iterate through all possible starting inventories
        # The index of these arrays corresponds to the starting inventory
        for i in range(storage_cap + 1):
            x[i], f[i] = optimizer_function(d_t, c_t, h_t, s_t, i, f_future)
        # Go one level deeper
        x_arr, f_arr, inv_arr = recursion_function(f, demand,
                                                   purchase_cost,
                                                   holding_cost, 
                                                   shortage_cost)
        # Work backwards up recursion stack to get decision values
        start_inv = inv_arr[-1]
        x_arr.append(x[start_inv])
        f_arr.append(x[start_inv])
        new_inv = int(max([0, start_inv + x[start_inv] - d_t]))
        inv_arr.append(new_inv)
    else:
        # At deepest level, find minimum future cost
        x, f = optimizer_function(d_t, c_t, h_t, s_t, 0, f_future)
        end_inv = int(max([x - d_t, 0]))

        x_arr = [x]
        f_arr = [f]
        inv_arr = [end_inv]
    return(x_arr, f_arr, inv_arr)


def optimizer_function(d_t, c_t, h_t, s_t, start_inv, f_future):
    # Set upper and lower x bounds based on constraints
    lb = 0
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
            if(final_inv >=0):
                future_cost = f_future[final_inv]
                f_arr.append(c_t * x + h_t * final_inv + future_cost)
            else:
                future_cost = f_future[0]
                f_arr.append(c_t * x + future_cost - final_inv * s_t) 
        f = min(f_arr)
        x = x_arr[f_arr.index(f)]
    return(x, f)


def main(T, plot = True):
    # Generate some data
    purchase_cost, holding_cost, demand, shortage_cost = generate_random_data(T)
    # Copy the lists since they will be modified in the recursion function
    demand_copy = demand.copy()
    holding_copy = holding_cost.copy()
    shortage_copy = shortage_cost.copy()
    purchase_copy = purchase_cost.copy()
    # For last period, future cost is 0
    f_future = [0] * (storage_cap + 1)
    # Run recursion function
    x_arr, fc_arr, inv_arr = recursion_function(f_future, demand,
                                                purchase_cost,
                                                holding_cost,shortage_cost)
    # Plot data
    if plot:
        weeks = list(range(1, T + 1))
        plt.figure()
        plt.plot(weeks, demand_copy, '--', label='demand')
        plt.plot(weeks, inv_arr, 'd', label='inventory')
        plt.plot(weeks, x_arr, '-', label='purchased')
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.title('Low Shortage Cost')
        plt.legend()
        plt.show()
    
    return(x_arr, fc_arr, inv_arr, purchase_copy, holding_copy, demand_copy, shortage_copy)

def testing():
    T_vals = [10, 20, 30]
    num_instances = 10
 
    array_shape = (num_instances, len(T_vals))
    runtime_array = np.empty(array_shape)
    inventory_array = np.empty(array_shape)
    max_inventory_array = np.empty(array_shape)
    holding_cost_array = np.empty(array_shape)
    purchase_cost_array = np.empty(array_shape)
    total_cost_array = np.empty(array_shape)
    shortage_cost_array = np.empty(array_shape)
 
    for j, T in enumerate(T_vals):
        for i in range(num_instances):
            start = time.time()
            x, fc, inv, pc, hc, d, sc = main(T, plot=False)
            end = time.time()
            
            #create an array of starting inventories based on the array of ending inventories
            st_inv = inv[-1:] + inv[0:-1]
            
            shorted = [0] * len(st_inv)
            for k in range(len(shorted)):
                shorted[k] = max(0, d[k] - st_inv[k] - x[k])

            # Convert to numpy arrays for easy manipulation
            x = np.asarray(x).reshape(T, 1)
            fc = np.asarray(fc).reshape(T, 1)
            inv = np.asarray(inv).reshape(T, 1)
            pc = np.asarray(pc).reshape(T, 1)
            hc = np.asarray(hc).reshape(T, 1)
            d = np.asarray(d).reshape(T, 1)
            sc = np.asarray(sc).reshape(T,1)
            shorted = np.asarray(shorted).reshape(T,1)
 
            # Calculate and store stats
            runtime_array[i, j] = (end - start) * 1000
            inventory_array[i, j] = np.mean(inv)
            max_inventory_array[i, j] = np.max(inv)
            holding_cost_array[i, j] = inv.T @ hc
            purchase_cost_array[i, j] = x.T @ pc
            shortage_cost_array[i, j] = shorted.T @ sc
            total_cost_array[i, j] = (holding_cost_array[i, j]
                                      + purchase_cost_array[i, j]
                                      + shortage_cost_array[i, j])
            
    plt.figure()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
 
    plt.subplot(2, 3, 1)
    plt.boxplot(inventory_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Units")
    plt.title("Average Inventory")
 
    plt.subplot(2, 3, 2)
    plt.boxplot(holding_cost_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Cost ($)")
    plt.title("Average Holding Cost")
 
    plt.subplot(2, 3, 3)
    plt.boxplot(purchase_cost_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Cost ($)")
    plt.title("Average Purchase Cost")
 
    plt.subplot(2, 3, 4)
    plt.boxplot(max_inventory_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Units")
    plt.title("Max Inventory Level")
 
    plt.subplot(2, 3, 5)
    plt.boxplot(total_cost_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Cost ($)")
    plt.title("Average Total Cost")
 
    plt.subplot(2, 3, 6)
    plt.boxplot(runtime_array, labels=T_vals)
    plt.xlabel("T")
    plt.ylabel("Time (ms)")
    plt.title("Runtime")
 
    for i in range(3):
        print(f"""T = {(i + 1) * 10}
            Inventory
                Avg: {np.mean(inventory_array[:, i])}
                Std: {np.std(inventory_array[:, i])}
            Max Inventory
                Avg: {np.mean(max_inventory_array[:, i])}
                Std: {np.std(max_inventory_array[:, i])}
            Holding Cost
                Avg: {np.mean(holding_cost_array[:, i])}
                Std: {np.std(holding_cost_array[:, i])}
            Purchase Cost
                Avg: {np.mean(purchase_cost_array[:, i])}
                Std: {np.std(purchase_cost_array[:, i])}
            Total Cost
                Avg: {np.mean(total_cost_array[:, i])}
                Std: {np.std(total_cost_array[:, i])}
            Runtime
                Avg: {np.mean(runtime_array[:, i])}
                Std: {np.std(runtime_array[:, i])}
        """)
 
    plt.show()
    return

if __name__ == "__main__":
    testing()
