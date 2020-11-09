\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: x_k_ij[0,1,0] = 0
 R1: x_k_ij[0,0,1] = 0
 R2: x_k_ij[0,2,0] = 0
 R3: x_k_ij[0,0,2] = 0
 R4: x_k_ij[0,3,0] = 0
 R5: x_k_ij[0,0,3] = 0
 R6: x_k_ij[0,4,0] = 0
 R7: x_k_ij[0,0,4] = 0
 R8: x_k_ij[0,0,1] = 1
 R9: x_k_ij[0,1,0] = 1
 R10: x_k_ij[0,0,2] = 1
 R11: x_k_ij[0,2,0] = 1
 R12: x_k_ij[0,0,3] = 1
 R13: x_k_ij[0,3,0] = 1
 R14: x_k_ij[0,0,4] = 1
 R15: x_k_ij[0,4,0] = 1
 R16: x_k_ij[1,1,0] = 0
 R17: x_k_ij[1,0,1] = 0
 R18: x_k_ij[1,2,0] = 0
 R19: x_k_ij[1,0,2] = 0
 R20: x_k_ij[1,3,0] = 0
 R21: x_k_ij[1,0,3] = 0
 R22: x_k_ij[1,4,0] = 0
 R23: x_k_ij[1,0,4] = 0
 R24: x_k_ij[1,0,1] = 1
 R25: x_k_ij[1,1,0] = 1
 R26: x_k_ij[1,0,2] = 1
 R27: x_k_ij[1,2,0] = 1
 R28: x_k_ij[1,0,3] = 1
 R29: x_k_ij[1,3,0] = 1
 R30: x_k_ij[1,0,4] = 1
 R31: x_k_ij[1,4,0] = 1
 capacity_vehicle_0: x_k_ij[0,1,4] + x_k_ij[0,1,0] + 3 x_k_ij[0,2,3]
   + 3 x_k_ij[0,2,0] + x_k_ij[0,3,2] + x_k_ij[0,3,4] + x_k_ij[0,3,0]
   + 2 x_k_ij[0,4,1] + 2 x_k_ij[0,4,3] + 2 x_k_ij[0,4,0] <= 4
 capacity_vehicle_1: x_k_ij[1,1,4] + x_k_ij[1,1,0] + 3 x_k_ij[1,2,3]
   + 3 x_k_ij[1,2,0] + x_k_ij[1,3,2] + x_k_ij[1,3,4] + x_k_ij[1,3,0]
   + 2 x_k_ij[1,4,1] + 2 x_k_ij[1,4,3] + 2 x_k_ij[1,4,0] <= 4
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
 time_precedence_from_node_1_to_0: - 11 x_k_ij[0,1,0] - 11 x_k_ij[1,1,0]
   + tau_i[0] - tau_i[1] >= -10
 time_precedence_from_node_2_to_3: - 2 x_k_ij[0,2,3] - 2 x_k_ij[1,2,3]
   - tau_i[2] + tau_i[3] >= 1
 time_precedence_from_node_2_to_0: - 7 x_k_ij[0,2,0] - 7 x_k_ij[1,2,0]
   + tau_i[0] - tau_i[2] >= -5
 time_precedence_from_node_3_to_2: - 8 x_k_ij[0,3,2] - 8 x_k_ij[1,3,2]
   + tau_i[2] - tau_i[3] >= -5
 time_precedence_from_node_3_to_4: - 15 x_k_ij[0,3,4] - 15 x_k_ij[1,3,4]
   - tau_i[3] + tau_i[4] >= 5
 time_precedence_from_node_3_to_0: - 9 x_k_ij[0,3,0] - 9 x_k_ij[1,3,0]
   + tau_i[0] - tau_i[3] >= -7
 time_precedence_from_node_4_to_1: - 10 x_k_ij[0,4,1] - 10 x_k_ij[1,4,1]
   + tau_i[1] - tau_i[4] >= -6
 time_precedence_from_node_4_to_3: - 28 x_k_ij[0,4,3] - 28 x_k_ij[1,4,3]
   + tau_i[3] - tau_i[4] >= -8
 time_precedence_from_node_4_to_0: - 15 x_k_ij[0,4,0] - 15 x_k_ij[1,4,0]
   + tau_i[0] - tau_i[4] >= -14
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
