from p_ERMlogl_bc import SPeq_solution
import numpy as np
import os

# flag = 0: Save solutions to compare to empirical results
# flag = 1: Save solutions at fixed alpha and epsilon by varying reg_strength

flag = 0

if flag == 0:
    reg_strengths = [1.0, 10, 100] # Values considered up to now: 1.0, 10, 100

    epsilons = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Values considered up to now: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    inf_alpha = 0
    sup_alpha = 50
    alphas = np.linspace(inf_alpha, sup_alpha, 501)

    # Initial values for m, q, sigma
    m0, q0, sigma0 = 0.1, 0.8, 3
    
    for reg_strength in reg_strengths:
        # Prepare the output directory only once per reg_strength
        base_dir = f'SP_solutions/to_compare_with_empirical_results/lambda={reg_strength}'
        os.makedirs(base_dir, exist_ok=True)

        for epsilon in epsilons:
            data = []
            for alpha in alphas:
                overlaps, converged = SPeq_solution(alpha, epsilon, reg_strength, m0, q0, sigma0)
                if converged:
                    # Update state for faster convergence in subsequent steps
                    m0, q0, sigma0, _, _, _ = overlaps
                    # Append data with pre-formatted structure for speed
                    data.append([alpha, epsilon, reg_strength, *overlaps])
                    # Save results
                    data_array = np.array(data, dtype=object) # Use object dtype for flexibility
                    filename = f"{base_dir}/p_ERMlogl_SP_solutions_epsilon={epsilon}_lambda={reg_strength}"
                    np.savetxt(filename, data_array, fmt='%s', delimiter=', ',
                               header="alpha, epsilon, lambda, m, q, sigma, m_hat, q_hat, sigma_hat")
elif flag == 1:
    alphas = [0.2, 0.4, 0.6, 0.8, 2, 4, 6, 8, 10] # Values considered up to now: 0.2, 0.4, 0.6, 0.8, 2, 4, 6, 8, 10

    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Values considered up to now: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    order_inf_reg_strength = -4
    order_sup_reg_strength = 2
    reg_strengths = np.flip(np.logspace(order_inf_reg_strength, order_sup_reg_strength, num=100))

    # Directory paths
    alpha_lt_1_dir = 'SP_solutions/txts_alpha_fixed/alpha < 1'
    os.makedirs(alpha_lt_1_dir, exist_ok=True)
    alpha_gt_1_dir = 'SP_solutions/txts_alpha_fixed/alpha > 1'
    os.makedirs(alpha_gt_1_dir, exist_ok=True)

    # Initial values for m, q, sigma
    m0, q0, sigma0 = 0.1, 0.8, 3

    for alpha in alphas:
        for epsilon in epsilons:
            # Determine file path based on alpha range and epsilon
            if alpha < 1:
                dir_path = f'{alpha_lt_1_dir}/alpha={alpha:.1f}'
                file_path = f'{dir_path}/p_ERMlogl_SP_solutions_alpha={alpha:.1f}_epsilon={epsilon}'
            else:
                dir_path = f'{alpha_gt_1_dir}/alpha={alpha}'
                file_path = f'{dir_path}/p_ERMlogl_SP_solutions_alpha={alpha}_epsilon={epsilon}'
            os.makedirs(dir_path, exist_ok=True)
            
            data = []
            for reg_strength in reg_strengths:
                overlaps, converged = SPeq_solution(alpha, epsilon, reg_strength, m0, q0, sigma0)
                if converged:
                    # Update state for faster convergence in subsequent steps
                    m0, q0, sigma0, _, _, _ = overlaps

                    # Append data with pre-formatted structure for speed
                    data.append([alpha, epsilon, reg_strength, *overlaps])
                    # Save results
                    data_array = np.array(data, dtype=object) # Use object dtype for flexibility
                    np.savetxt(file_path, data_array, fmt='%s', delimiter=', ',
                               header="alpha, epsilon, lambda, m, q, sigma, m_hat, q_hat, sigma_hat")
