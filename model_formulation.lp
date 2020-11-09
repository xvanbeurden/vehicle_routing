\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4 x_k_ij[0,1,4] + x_k_ij[0,1,0] + 3 x_k_ij[0,2,3] + 2 x_k_ij[0,2,0]
   + 3 x_k_ij[0,3,2] + 20 x_k_ij[0,3,4] + 2 x_k_ij[0,3,0] + 4 x_k_ij[0,4,1]
   + 20 x_k_ij[0,4,3] + x_k_ij[0,4,0] + x_k_ij[0,0,1] + 2 x_k_ij[0,0,2]
   + 2 x_k_ij[0,0,3] + x_k_ij[0,0,4] + 4 x_k_ij[1,1,4] + x_k_ij[1,1,0]
   + 3 x_k_ij[1,2,3] + 2 x_k_ij[1,2,0] + 3 x_k_ij[1,3,2] + 20 x_k_ij[1,3,4]
   + 2 x_k_ij[1,3,0] + 4 x_k_ij[1,4,1] + 20 x_k_ij[1,4,3] + x_k_ij[1,4,0]
   + x_k_ij[1,0,1] + 2 x_k_ij[1,0,2] + 2 x_k_ij[1,0,3] + x_k_ij[1,0,4]
Subject To
 vehicle_0_leaves_depot: x_k_ij[0,0,1] + x_k_ij[0,0,2] + x_k_ij[0,0,3]
   + x_k_ij[0,0,4] = 1
 vehicle_0_enters_depot: x_k_ij[0,1,0] + x_k_ij[0,2,0] + x_k_ij[0,3,0]
   + x_k_ij[0,4,0] = 1
 vehicle_1_leaves_depot: x_k_ij[1,0,1] + x_k_ij[1,0,2] + x_k_ij[1,0,3]
   + x_k_ij[1,0,4] = 1
 vehicle_1_enters_depot: x_k_ij[1,1,0] + x_k_ij[1,2,0] + x_k_ij[1,3,0]
   + x_k_ij[1,4,0] = 1
 visit_customer_1_once: x_k_ij[0,4,1] + x_k_ij[0,0,1] + x_k_ij[1,4,1]
   + x_k_ij[1,0,1] = 1
 visit_customer_2_once: x_k_ij[0,3,2] + x_k_ij[0,0,2] + x_k_ij[1,3,2]
   + x_k_ij[1,0,2] = 1
 visit_customer_3_once: x_k_ij[0,2,3] + x_k_ij[0,4,3] + x_k_ij[0,0,3]
   + x_k_ij[1,2,3] + x_k_ij[1,4,3] + x_k_ij[1,0,3] = 1
 visit_customer_4_once: x_k_ij[0,1,4] + x_k_ij[0,3,4] + x_k_ij[0,0,4]
   + x_k_ij[1,1,4] + x_k_ij[1,3,4] + x_k_ij[1,0,4] = 1
 same_vehicle_at_node_1: - x_k_ij[0,1,4] - x_k_ij[0,1,0] + x_k_ij[0,4,1]
   + x_k_ij[0,0,1] = 0
 same_vehicle_at_node_1: - x_k_ij[1,1,4] - x_k_ij[1,1,0] + x_k_ij[1,4,1]
   + x_k_ij[1,0,1] = 0
 same_vehicle_at_node_2: - x_k_ij[0,2,3] - x_k_ij[0,2,0] + x_k_ij[0,3,2]
   + x_k_ij[0,0,2] = 0
 same_vehicle_at_node_2: - x_k_ij[1,2,3] - x_k_ij[1,2,0] + x_k_ij[1,3,2]
   + x_k_ij[1,0,2] = 0
 same_vehicle_at_node_3: x_k_ij[0,2,3] - x_k_ij[0,3,2] - x_k_ij[0,3,4]
   - x_k_ij[0,3,0] + x_k_ij[0,4,3] + x_k_ij[0,0,3] = 0
 same_vehicle_at_node_3: x_k_ij[1,2,3] - x_k_ij[1,3,2] - x_k_ij[1,3,4]
   - x_k_ij[1,3,0] + x_k_ij[1,4,3] + x_k_ij[1,0,3] = 0
 same_vehicle_at_node_4: x_k_ij[0,1,4] + x_k_ij[0,3,4] - x_k_ij[0,4,1]
   - x_k_ij[0,4,3] - x_k_ij[0,4,0] + x_k_ij[0,0,4] = 0
 same_vehicle_at_node_4: x_k_ij[1,1,4] + x_k_ij[1,3,4] - x_k_ij[1,4,1]
   - x_k_ij[1,4,3] - x_k_ij[1,4,0] + x_k_ij[1,0,4] = 0
 capacity_vehicle_0: x_k_ij[0,1,4] + x_k_ij[0,1,0] + x_k_ij[0,2,3]
   + x_k_ij[0,2,0] + x_k_ij[0,3,2] + x_k_ij[0,3,4] + x_k_ij[0,3,0]
   + x_k_ij[0,4,1] + x_k_ij[0,4,3] + x_k_ij[0,4,0] <= 100
 capacity_vehicle_1: x_k_ij[1,1,4] + x_k_ij[1,1,0] + x_k_ij[1,2,3]
   + x_k_ij[1,2,0] + x_k_ij[1,3,2] + x_k_ij[1,3,4] + x_k_ij[1,3,0]
   + x_k_ij[1,4,1] + x_k_ij[1,4,3] + x_k_ij[1,4,0] <= 100
 lower_bound_visiting_time_node_0: tau_i[0] >= 0
 upper_bound_visiting_time_node_0: tau_i[0] <= 20
 lower_bound_visiting_time_node_1: tau_i[1] >= 8
 upper_bound_visiting_time_node_1: tau_i[1] <= 10
 lower_bound_visiting_time_node_2: tau_i[2] >= 2
 upper_bound_visiting_time_node_2: tau_i[2] <= 5
 lower_bound_visiting_time_node_3: tau_i[3] >= 6
 upper_bound_visiting_time_node_3: tau_i[3] <= 7
 lower_bound_visiting_time_node_4: tau_i[4] >= 12
 upper_bound_visiting_time_node_4: tau_i[4] <= 14
 time_precedence_from_node_1_to_4: - 2 x_k_ij[0,1,4] - 2 x_k_ij[1,1,4]
   - tau_i[1] + tau_i[4] >= 2
 time_precedence_from_node_2_to_3: - 2 x_k_ij[0,2,3] - 2 x_k_ij[1,2,3]
   - tau_i[2] + tau_i[3] >= 1
 time_precedence_from_node_3_to_2: - 8 x_k_ij[0,3,2] - 8 x_k_ij[1,3,2]
   + tau_i[2] - tau_i[3] >= -5
 time_precedence_from_node_3_to_4: - 15 x_k_ij[0,3,4] - 15 x_k_ij[1,3,4]
   - tau_i[3] + tau_i[4] >= 5
 time_precedence_from_node_4_to_1: - 10 x_k_ij[0,4,1] - 10 x_k_ij[1,4,1]
   + tau_i[1] - tau_i[4] >= -6
 time_precedence_from_node_4_to_3: - 28 x_k_ij[0,4,3] - 28 x_k_ij[1,4,3]
   + tau_i[3] - tau_i[4] >= -8
 time_precedence_from_node_0_to_1: - 13 x_k_ij[0,0,1] - 13 x_k_ij[1,0,1]
   - tau_i[0] + tau_i[1] >= -12
 time_precedence_from_node_0_to_2: - 20 x_k_ij[0,0,2] - 20 x_k_ij[1,0,2]
   - tau_i[0] + tau_i[2] >= -18
 time_precedence_from_node_0_to_3: - 16 x_k_ij[0,0,3] - 16 x_k_ij[1,0,3]
   - tau_i[0] + tau_i[3] >= -14
 time_precedence_from_node_0_to_4: - 9 x_k_ij[0,0,4] - 9 x_k_ij[1,0,4]
   - tau_i[0] + tau_i[4] >= -8
Bounds
Binaries
 x_k_ij[0,1,4] x_k_ij[0,1,0] x_k_ij[0,2,3] x_k_ij[0,2,0] x_k_ij[0,3,2]
 x_k_ij[0,3,4] x_k_ij[0,3,0] x_k_ij[0,4,1] x_k_ij[0,4,3] x_k_ij[0,4,0]
 x_k_ij[0,0,1] x_k_ij[0,0,2] x_k_ij[0,0,3] x_k_ij[0,0,4] x_k_ij[1,1,4]
 x_k_ij[1,1,0] x_k_ij[1,2,3] x_k_ij[1,2,0] x_k_ij[1,3,2] x_k_ij[1,3,4]
 x_k_ij[1,3,0] x_k_ij[1,4,1] x_k_ij[1,4,3] x_k_ij[1,4,0] x_k_ij[1,0,1]
 x_k_ij[1,0,2] x_k_ij[1,0,3] x_k_ij[1,0,4]
End
