# Sets up and solves a vehicle routing problem.

# Imports.
import numpy as np
from gurobipy import Model, GRB, LinExpr  # How to get this module on your computer?

###############
# Model setup #
###############

# Import the data needed to solve this problem...

# Define the objective function.


#############
# Variables #
#############

# Define the decision variables.
# x_k_ij: vehicle k goes from i to j.

###############
# Constraints #
###############

# Each vehicle must leave the depot.


# Each vehicle must return to depot.


# Each customer must be visited once.


# If a vehicle visits a customer, the same vehicle must leave that customer.


# Each vehicle has a maximum capacity.


# Each customer must be visited within a certain time window.


# Optional: Buy extra vehicles.
# Optional: Time it takes to drop off a packages.
# Optional: Pickup packages and bring them back to the depot.


#####################
# Solving the model #
#####################

# Export is as an LP file.

# Optimize.
