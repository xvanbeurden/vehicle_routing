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
Q = 100         # Max capacity per vehicle.
n_vehicles = 2  # Number of vehicles.
n_nodes = 5     # Number of nodes.  # TODO: read from data.

# Import the data needed to solve this problem.
data = pd.read_excel("data_simplemodel.xlsx")

links = data[["From", "To", "Distance"]].dropna().astype(int)

q_i = np.array(data["Gewicht"].dropna().astype(int).tolist())
e_i = np.array(data["time_begin"].dropna().astype(int).tolist())
l_i = np.array(data["time_end"].dropna().astype(int).tolist())
p_i = np.array(data["processing_time"].dropna().astype(int).tolist())

#############
# Variables #
#############

# Define the decision variables.
i_to_j = [(i, j) for i, j in links[["From", "To"]].itertuples(index=False)]
x_k_ij = model.addVars(n_vehicles, i_to_j, vtype=GRB.BINARY, name="x_k_ij")  # vehicle k goes from i to j.
tau_i = model.addVars(n_nodes, vtype=GRB.CONTINUOUS, name="tau_i")           # Start of service at customer i.


###############
# Constraints #
###############

# Each vehicle must leave and return to the depot.
for k in range(n_vehicles):
    leave_once = LinExpr()
    enter_once = LinExpr()

    for i, j in data[['From', 'To']].itertuples(index=False):
        if i == 0:
            leave_once += x_k_ij[k, i, j]

        if j == 0:
            enter_once += x_k_ij[k, i, j]

    model.addConstr(lhs=leave_once, sense = GRB.EQUAL, rhs=1, name=f"vehicle_{k}_leaves_depot")
    model.addConstr(lhs=enter_once, sense = GRB.EQUAL, rhs=1, name=f"vehicle_{k}_enters_depot")


# Each customer must be visited once.
for n in range(1, n_nodes):
    enters_once = LinExpr()

    for i, j in data[['From', 'To']].itertuples(index=False):

        if j == n:
            for k in range(n_vehicles):
                enters_once += x_k_ij[k, i, j]

    model.addConstr(lhs=enters_once, sense=GRB.EQUAL, rhs=1, name=f"visit_customer_{n}_once")


# If a vehicle visits a customer, the same vehicle must leave that customer.
for n in range(1, n_nodes):
    for k in range(n_vehicles):

        flow = LinExpr()
        for i, j in data[['From', 'To']].itertuples(index=False):

            if j == n:
                flow += x_k_ij[k, i, j]

            if i == n:
                flow -= x_k_ij[k, i, j]

        model.addConstr(lhs=flow, sense=GRB.EQUAL, rhs=0, name=f"same_vehicle_at_node_{n}")


# Each vehicle has a maximum capacity.
for k in range(n_vehicles):
    used_capacity_k = LinExpr()

    # If this vehicle travels from i to j we add the demand of customer i to the used capacity of this vehicle.
    for i, j in data[['From', 'To']].itertuples(index=False):
        used_capacity_k += q_i[i] * x_k_ij[k, i, j]

    model.addConstr(used_capacity_k <= Q, name=f"capacity_vehicle_{k}")


# Each customer must be visited within a certain time window.
# This has two subconstraints. The first one is that a customer must be visited in their specified time window.
for i in range(n_nodes):
    model.addConstr(tau_i[i] >= e_i[i], name=f"lower_bound_visiting_time_node_{i}")
    model.addConstr(tau_i[i] <= l_i[i], name=f"upper_bound_visiting_time_node_{i}")

# The second subconstraint is a time precedence constraint. If we travel from i to j, we can't arrive sooner than the
# time it takes to drop the package at i and travel from i to j.
for i, j, t in links.itertuples(index=False):

    # We must arrive at the depot at the end so we should skip it here.
    if j == 0:
        continue

    # We use the big M method so this constraint is always satisfied if we don't travel from i to j. We want
    # M to be big but not too big, so we calculate its minimum value.
    M = max(0, l_i[i] + p_i[i] + t - e_i[j])

    # Check if we travel from i to j in any vehicle.
    x = 0
    for k in range(n_vehicles):
        x += x_k_ij[k, i, j]

    model.addConstr(tau_i[j] >= tau_i[i] + p_i[i] + t - (1 - x) * M,
                    name=f"time_precedence_from_node_{i}_to_{j}")

# Optional: Buy extra vehicles.
# Optional: Pickup packages and bring them back to the depot.


#####################
# Solving the model #
#####################

# Define the objective function.
obj = LinExpr()
for i, j, t in links.itertuples(index=False):
    for k in range(n_vehicles):
        obj += x_k_ij[k, i, j] * t

model.setObjective(obj, GRB.MINIMIZE)

# Everything is defined, so update the model.
model.update()

# Export is as an LP file.
model.write('model_formulation.lp')

# Optimize.
model.optimize()

# Saving our solution in the form [name of variable, value of variable]
sol = []
for v in model.getVars():
     sol.append([v.varName, v.x])
