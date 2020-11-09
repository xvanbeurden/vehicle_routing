# Sets up and solves a vehicle routing problem.

# Imports.
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, LinExpr

###############
# Model setup #
###############

# Initialize the model.
model = Model()

# Parameters.
Q = 4           # Max capacity per vehicle.
n_vehicles = 2  # Number of vehicles.
n_nodes = 5     # Number of nodes.

# Import the data needed to solve this problem.
data = pd.read_excel("data_simplemodel.xlsx")

links = data[["From", "To", "Distance"]].dropna().astype(int)

q_i = data[["Station", "Gewicht"]].dropna().astype(int).set_index("Station")
e_i = data[["Station", "time_begin"]].dropna().astype(int).set_index("Station")
l_i = data[["Station", "time_end"]].dropna().astype(int).set_index("Station")
p_i = data[["Station", "processing_time"]].dropna().astype(int).set_index("Station")

#############
# Variables #
#############

# Define the decision variables.
l = [(start, stop) for start, stop in links[["From", "To"]].itertuples(index=False)]
x_k_ij = model.addVars(n_vehicles, l, vtype=GRB.BINARY)  # vehicle k goes from i to j, binary.
tau_i = model.addVars(n_nodes, lb=0, vtype=GRB.CONTINUOUS)  # Start of service at customer i, continuous.


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
for i, j, t in links.itertuples(index=False):
    if i == 0 or j == 0:
        continue
    else:
        # We use the big M method so this constraint is always satisfied if we don't travel from i to j. We want
        # M to be big but not too big, so we calculate its minimum value.
        M = max(0, l_i[i] + p_i[i] + t - e_i[j])

        # Check if we travel from i to j in any vehicle.
        x = 0
        for k in range(n_vehicles):
            x += x_k_ij[k, i, j]

        model.addConstr(tau_i[j] >= tau_i[i] + p_i[i] + t - (1 - x) * M,
                        name=f"time_precedence_from_{i}_to_{j}")

# Optional: Buy extra vehicles.
# Optional: Pickup packages and bring them back to the depot.


#####################
# Solving the model #
#####################

# Define the objective function.

# Export is as an LP file.

# Optimize.
