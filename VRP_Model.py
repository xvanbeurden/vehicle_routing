# Sets up and solves a vehicle routing problem.

# Imports.
import numpy as np
from gurobipy import Model, GRB, LinExpr  # How to get this module on your computer?

###############
# Model setup #
###############

# Import the data needed to solve this problem.

# Temporary hard-coded parameters.
q_i = [1, 1, 1, 1]  # Demand for customer i.
Q = 2  # Max capacity per vehicle.
n_vehicles = 2  # Number of vehicles.
n_customers = 4  # Number of customers.

# Define the objective function.


#############
# Variables #
#############

# Define the decision variables.
x_k_ij = np.zeros((n_vehicles, n_customers, n_customers))  # vehicle k goes from i to j.

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
    used_capacity_k = 0

    # Check if this vehicle goes from i to j, if so, add the demand of customer i to the used capacity of this vehicle.
    for i in range(n_customers):
        for j in range(n_customers):
            if i == j:
                continue
            else:
                used_capacity_k += q_i[i] * x_k_ij[k][i][j]

    LHS = used_capacity_k
    RHS = Q

    # TODO: add contraint with gurobipy.


# Each customer must be visited within a certain time window.


# Optional: Buy extra vehicles.
# Optional: Time it takes to drop off a packages.
# Optional: Pickup packages and bring them back to the depot.


#####################
# Solving the model #
#####################

# Export is as an LP file.

# Optimize.
