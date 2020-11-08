# Sets up and solves a vehicle routing problem.

# Imports.
import numpy as np
from gurobipy import Model, GRB, LinExpr

###############
# Model setup #
###############

# Initialize the model.
model = Model()

# Import the data needed to solve this problem.

# Temporary hard-coded parameters.
t_ij = np.array([[1, 1], [1, 1]])  # Travel time between customer i and j. TODO: What if i-j has no connection?
q_i = np.array([1, 1])             # Demand for customer i.
Q = 2                              # Max capacity per vehicle.
n_vehicles = 2                     # Number of vehicles.
n_customers = 2                    # Number of customers.
e_i = np.array([1, 1])             # Earliest we can visit customer i.
l_i = np.array([2, 2])             # Latest we can visit customer i.
p_i = np.array([0, 0])             # Processing time at customer i.

# Define the objective function.


#############
# Variables #
#############

# Define the decision variables.
# See: https://www.gurobi.com/documentation/9.0/refman/py_model_addvars.html
# TODO: What if i-j has no connection?
x_k_ij = model.addVars(n_vehicles, n_customers, n_customers, vtype=GRB.BINARY)  # vehicle k goes from i to j, binary.
tau_i = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS)                  # Start of service at customer i, continuous.


###############
# Constraints #
###############

# Each vehicle must leave the depot.


# Each vehicle must return to depot.


# Each customer must be visited once.


# If a vehicle visits a customer, the same vehicle must leave that customer.


# Each vehicle has a maximum capacity.
# For each vehicle we take the product of the demand for customer i with
for k in range(n_vehicles):
    used_capacity_k = LinExpr()

    # Check if this vehicle goes from i to j, if so, add the demand of customer i to the used capacity of this vehicle.
    for i in range(n_customers):
        for j in range(n_customers):
            if i == j:
                continue
            else:
                used_capacity_k += q_i[i] * x_k_ij[k, i, j]

    model.addConstr(used_capacity_k <= Q, name=f"capacity_vehicle_{k}")


# Each customer must be visited within a certain time window.
# This has two subconstraints. The first one is that a customer must be visited in their specified time window.
for i in range(n_customers):
    model.addConstr(tau_i[i] >= e_i[i], name=f"lower_bound_visiting_time_customer_{i}")
    model.addConstr(tau_i[i] <= l_i[i], name=f"upper_bound_visiting_time_customer_{i}")

# The second subconstraint is a time precedence constraint. If we travel from i to j, we can't arrive sooner than the
# time it takes to drop the package at i and travel from i to j.
for i in range(n_customers):
    for j in range(n_customers):
        if i == j:
            continue

        else:
            # We use the big M method so this constraint is always satisfied if we don't travel from i to j. We want
            # M to be big but not too big, so we calculate its minimum value.
            M = max(0, l_i[i] + p_i[i] + t_ij[i, j] - e_i[j])

            # Check if we travel from i to j in any vehicle.
            x = 0
            for k in range(n_vehicles):
                x += x_k_ij[k, i, j]

            model.addConstr(tau_i[j] >= tau_i[i] + p_i[i] + t_ij[i, j] - (1 - x) * M,
                            name=f"time_precedence_from_{i}_to_{j}")

# Optional: Buy extra vehicles.
# Optional: Pickup packages and bring them back to the depot.


#####################
# Solving the model #
#####################

# Export is as an LP file.

# Optimize.
