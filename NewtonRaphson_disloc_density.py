
import numpy as np 
import matplotlib.pyplot as plt

#Newton-Raphson solver for changed variable u = sqrt(rho) 
def rho_solver(k1, k2, d, sigma, r_p, r_gb, init_u=1, tol=1e-10, max_iter = 1000):
    
    u = init_u
    converged = False

    for i in range(max_iter):
    #function f(u) and df(u)/du   
        f = (k1 * u + k2/d)*np.sinh(sigma - u - 1/d) - (r_p * u**4 + r_gb * u**2/d)
        df = k1 * np.sinh(sigma - u - 1/d) - (k1 * u + k2/d)*np.cosh(sigma - u - 1/d) - (4 * r_p * u**3 + 2 * r_gb * u/d)

        if abs(df) < 1e-15:
            print("Derivative too small")
            return None
        
        u_new = u - f/df
       

        if abs(u_new - u) < tol:
            u = u_new
            converged = True
            break

        u = u_new

    if not converged:
        print("Solver did not converge")
        return None

    #check residual so we now that it is a true solution 
    rho_sol = u**2 

    res =  (k1 * np.sqrt(rho_sol) + k2/d)*np.sinh(sigma - np.sqrt(rho_sol) - 1/d) - (r_p * rho_sol**2 + r_gb * rho_sol/d)

    print(f"dislocation density solution: {rho_sol:.6e}, residual: {res:.6e}, iterations: {i+1}")
    print(f"u solution: {u:.6e}, residual: {res:.6e}, iterations: {i+1}")

    return rho_sol 

#parameters from table (subject to change with feedback)

params = {
    "sigma": 2.345,
    "d": 131.12,
    "k1": 4686.40,
    "k2": 4499.63,
    "r_p": 1,
    "r_gb": 1
}

rho_solution = rho_solver(**params)

df_val = params["k1"] * np.sinh(params["sigma"] - np.sqrt(rho_solution) - 1/params["d"]) - (params["k1"] * np.sqrt(rho_solution) + params["k2"]/params["d"])*np.cosh(params["sigma"] - np.sqrt(rho_solution) - 1/params["d"]) - (4 * params["r_p"] * rho_solution**(3/2) + 2 * params["r_gb"] * np.sqrt(rho_solution)/params["d"])


print(df_val)