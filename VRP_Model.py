"""
Constraints:
- Each vehicle must leave depot.
- Each vehicle must return to depot.
- Each customer must be visited once.
- If a vehicle visits a customer the same vehicle must leave that customer.
- Vehicle capacity.
- Time windows

Optional constraints:
- Buy extra vehicles.
- Delivery time (time it takes to drop off package).
- Pickup packages and bring them back to depot.

Decision variables:
- x_k_ij: vehicle k goes from i to j.

"""
