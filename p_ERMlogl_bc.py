import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import fmin
from numba import jit

@jit
def f_out(y, omega, V, f0 = 0.0, tol = 1e-4, max_iter = 10000, damping = 0.1):
    ''' Solve a fixed point iteration to find the value of f_out for given y, omega, V '''
    fs = [f0]
    for _ in range(max_iter):
        f = y / (1 + np.exp(y * (V * f0 + omega)))
        err_tol = np.abs(f - f0)
        if err_tol < tol:
            return f

        f0 = (1 - damping) * f0 + damping * f
        fs.append(f0)
        if len(fs) > 2:
            if (f0 - fs[-2]) * (fs[-2] - fs[-3]) < 0: # fs[-1] == f0
                damping *= 0.1

    print(f"Fixed point for f_out not found. Last err_tol = {err_tol} (tol = {tol})")
    return None

@jit
def partial_f_out(y, omega, V, partial_f0 = 0.0, tol=1e-4, max_iter = 10000, damping = 0.1):
    ''' Solve a fixed point iteration to find the value of the partial derivative of f_out wrt to omega for given y, omega, V '''
    partial_fs = [partial_f0]
    for _ in range(max_iter):
        partial_f = - f_out(y, omega, V) * y * (V * partial_f0 + 1) / (1 + np.exp(- y * (V * f_out(y, omega, V) + omega)))
        err_tol = np.abs(partial_f - partial_f0)
        if err_tol < tol:
            return partial_f
        
        partial_f0 = (1 - damping) * partial_f0 + damping * partial_f
        partial_fs.append(partial_f0)
        if len(partial_fs) > 2:
            if (partial_f0 - partial_fs[-2]) * (partial_fs[-2] - partial_fs[-3]) < 0: # partial_fs[-1] == partial_f0
                damping *= 0.1

    print(f"Fixed point for partial_f_out not found. Last err_tol = {err_tol} (tol = {tol})")
    return None

def F_m(reg_strength, m_hat, sigma_hat):
    ''' Rhs of the SP equation for m '''
    return m_hat / (reg_strength + sigma_hat)

def F_q(reg_strength, m_hat, q_hat, sigma_hat):
    ''' Rhs of the SP equation for q '''
    return (m_hat**2 + q_hat) / (reg_strength + sigma_hat)**2

def F_sigma(reg_strength, sigma_hat):
    ''' Rhs of the SP equation for sigma '''
    return 1 / (reg_strength + sigma_hat)

def F_m_hat(alpha, epsilon, m, q, sigma):
    ''' Rhs of the SP equation for m_hat '''
    return alpha * (1 - epsilon) / (2 * np.pi) * np.sqrt(q / (q - m**2)) * quad(lambda t: np.exp(-q * t**2 / (2 * (q - m**2))) * (f_out(1, np.sqrt(q) * t, sigma) - f_out(-1, np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0]

def F_q_hat(alpha, epsilon, m, q, sigma):
    ''' Rhs of the SP equation for q_hat '''
    return alpha * epsilon / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (f_out(-1, np.sqrt(q) * t, sigma)**2 + f_out(1, np.sqrt(q) * t, sigma)**2), -np.inf, np.inf)[0] + alpha * (1 - epsilon) / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (1 + erf(m * t / np.sqrt(2 * (q - m**2)))) * (f_out(-1, -np.sqrt(q) * t, sigma)**2 + f_out(1, np.sqrt(q) * t, sigma)**2), -np.inf, np.inf)[0]

def F_sigma_hat(alpha, epsilon, m, q, sigma):
    ''' Rhs of the SP equation for sigma_hat '''
    return - alpha * epsilon / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (partial_f_out(-1, np.sqrt(q) * t, sigma) + partial_f_out(1, np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0] - alpha * (1 - epsilon) / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (1 + erf(m * t / np.sqrt(2 * (q - m**2)))) * (partial_f_out(-1, -np.sqrt(q) * t, sigma) + partial_f_out(1, np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0]

def state_evolution_eq(overlaps, alpha, epsilon, reg_strength):
    ''' Perform one step of the state evolution and return the result '''
    m, q, sigma, _, _, _ = overlaps

    m_hat_new = F_m_hat(alpha, epsilon, m, q, sigma)
    q_hat_new = F_q_hat(alpha, epsilon, m, q, sigma)
    sigma_hat_new = F_sigma_hat(alpha, epsilon, m, q, sigma)

    m_new = F_m(reg_strength, m_hat_new, sigma_hat_new)
    q_new = F_q(reg_strength, m_hat_new, q_hat_new, sigma_hat_new)
    sigma_new = F_sigma(reg_strength, sigma_hat_new)

    new_overlaps = np.array([m_new, q_new, sigma_new, m_hat_new, q_hat_new, sigma_hat_new])

    return new_overlaps

def SPeq_solution(alpha, epsilon, reg_strength, m0 = 0.1, q0 = 0.8, sigma0 = 3, tol = 1e-3, damping = True, delta = 0.9, max_iter = 5000):
    ''' Solve a fixed point iteration to find the SP overlaps and hat variables for given alpha, epsilon and reg_strength \n
        Set damping = True to apply damping '''
    
    m_hat0 = F_m_hat(alpha, epsilon, m0, q0, sigma0)
    q_hat0 = F_q_hat(alpha, epsilon, m0, q0, sigma0)
    sigma_hat0 = F_sigma_hat(alpha, epsilon, m0, q0, sigma0)

    overlaps = np.array([m0, q0, sigma0, m_hat0, q_hat0, sigma_hat0])

    for _ in range(max_iter):
        new_overlaps = state_evolution_eq(overlaps, alpha, epsilon, reg_strength)
        err_tol = np.linalg.norm(new_overlaps - overlaps)

        if err_tol < tol:
            print(f"Convergence reached for alpha = {alpha}, epsilon = {epsilon}, lambda = {reg_strength}")
            return overlaps, True
        
        if damping == True:
            overlaps = (1 - delta) * overlaps + delta * new_overlaps
        else:
            overlaps = new_overlaps
    
    print(f"Convergence not reached for alpha = {alpha}, epsilon = {epsilon}, lambda = {reg_strength}, last err_tol = {err_tol} (tol = {tol})")
    return overlaps, False

def training_loss(alpha, epsilon, reg_strength, m, q, sigma):
    ''' Compute training loss for given alpha, epsilon, reg_strength, m, q, sigma '''
    def logistic_envelope(y, omega, V, z0 = 0.5):
        return fmin(lambda z: np.log(1 + np.exp(-y * z)) + (z - omega)**2 / (2 * V), x0=z0, full_output=True, disp=False)[1]

    if epsilon == 0.0:
        return (reg_strength * q / 2 - (q - m**2) / (2 * sigma) + alpha / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (1 + erf(m * t / np.sqrt(2 * (q - m**2)))) * (logistic_envelope(1, np.sqrt(q) * t, sigma) + logistic_envelope(-1, -np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0]) / alpha
    elif epsilon == 1.0:
        return (reg_strength * q / 2 - (q - m**2) / (2 * sigma) + alpha / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (logistic_envelope(-1, np.sqrt(q) * t, sigma) + logistic_envelope(1, np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0]) / alpha
    else:
        return (reg_strength * q / 2 - (q - m**2) / (2 * sigma) + alpha * epsilon / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (logistic_envelope(-1, np.sqrt(q) * t, sigma) + logistic_envelope(1, np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0] + alpha * (1 - epsilon) / (2 * np.sqrt(2 * np.pi)) * quad(lambda t: np.exp(-t**2/2) * (1 + erf(m * t / np.sqrt(2 * (q - m**2)))) * (logistic_envelope(1, np.sqrt(q) * t, sigma) + logistic_envelope(-1, -np.sqrt(q) * t, sigma)), -np.inf, np.inf)[0]) / alpha

def training_error_onrandomlylabelled(q, sigma):
    ''' Compute memorization error for given q, sigma '''
    def logistic_proximal(y, omega, V, z0 = 0.5, tol = 1e-5, max_iter = 5000):
        for _ in range(max_iter):
            z = V * y / (np.exp(y * z0) + 1) + omega
            err_tol = np.abs(z - z0)
            if err_tol < tol:
                return z
            z0 = z
        print(f"Fixed point for logistic_proximal not found, last err_tol = {err_tol} (tol = {tol})")
        return None

    return quad(lambda t: np.exp(-t**2/2) * (np.heaviside(logistic_proximal(-1, np.sqrt(q) * t, sigma), 0) + np.heaviside(- logistic_proximal(1, np.sqrt(q) * t, sigma), 0)), -np.inf, np.inf)[0] / (2 * np.sqrt(2 * np.pi))

def generalization_error(m, q):
    ''' Compute generalization error for given m, q '''
    if m == 0:
        return 1 / 2
    else:
        return np.arccos(m / np.sqrt(q)) / np.pi

