"""
Generate figures for PublicationE paper
Method Comparison: Regressor vs Traditional with Sparse Data
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/rodo/1Paper/PublicationE')
from rbf_analytical import RBFAnalytical
from duffing_regressor_rbf import solve_duffing_regressor, true_spring_force
import time

# Matplotlib configuration
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Physical parameters
D = 0.2
A_SPRING = 1.0
B_SPRING = 0.5
A_FORCE = 0.8
OMEGA = 1.2

def solve_duffing_rk45(t_span, y0, v0):
    """Solve with RK45 (reference solution)"""
    def duffing_ode(t, state):
        y, v = state
        f_spring = true_spring_force(y)
        f_ext = A_FORCE * np.cos(OMEGA * t)
        dvdt = -D * v - f_spring + f_ext
        return [v, dvdt]

    sol = solve_ivp(duffing_ode, t_span, [y0, v0],
                    method='RK45', dense_output=True,
                    max_step=0.01, rtol=1e-10, atol=1e-12)
    return sol

def train_rbf(y_data, f_data, n_centers):
    """Train RBF to approximate f(y)"""
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = (y_max - y_min) / (2 * n_centers)

    # Design matrix
    distances = y_data.reshape(-1, 1) - centers.reshape(1, -1)
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi_aug = np.hstack([phi, np.ones((len(y_data), 1))])

    # Least squares
    weights = np.linalg.lstsq(phi_aug, f_data, rcond=None)[0]

    rbf = RBFAnalytical(centers, sigma, weights)
    return rbf

def traditional_method(t_data, y_data, n_centers):
    """Traditional method: solve for unknowns + RBF"""
    # Calculate numerical derivatives
    dt = np.diff(t_data)
    v = np.diff(y_data) / dt
    v_ext = np.concatenate(([v[0]], v))

    dv = np.diff(v_ext) / dt
    dv_ext = np.concatenate(([dv[0]], dv))

    # Solve for f(y)
    f_despeje = -dv_ext - D * v_ext + A_FORCE * np.cos(OMEGA * t_data)

    # Train RBF
    rbf = train_rbf(y_data, f_despeje, n_centers)
    return rbf, f_despeje

print("="*70)
print("GENERATING FIGURES FOR PUBLICATION E")
print("="*70)

# ==============================================================================
# FIGURE 1: Comparison with N=10 points (very sparse data)
# ==============================================================================
print("\n[1/3] Generating figure 1: Comparison N=10...")

N = 10
T_MAX = 15.0
t_data = np.linspace(0, T_MAX, N)
y0, v0 = 0.5, 0.0

# Reference solution
sol_ref = solve_duffing_rk45([0, T_MAX], y0, v0)
t_dense = np.linspace(0, T_MAX, 500)
y_ref = sol_ref.sol(t_dense)[0]

# Experimental data
y_data = sol_ref.sol(t_data)[0]
f_true = true_spring_force(y_data)

# Traditional method
rbf_trad, f_despeje = traditional_method(t_data, y_data, n_centers=3)
y_trad = solve_duffing_regressor(rbf_trad, t_dense, y0, sol_ref.sol(t_data)[1, 0])

# Regressor method
rbf_reg = train_rbf(y_data, f_true, n_centers=3)
y_reg = solve_duffing_regressor(rbf_reg, t_dense, y0, v0)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Subplot 1: Time solutions
ax = axes[0, 0]
ax.plot(t_data, y_data, 'ko', markersize=8, label='Data (N=10)', zorder=5)
ax.plot(t_dense, y_ref, 'k-', linewidth=2, label='Reference RK45', alpha=0.7)
ax.plot(t_dense, y_trad, 'r--', linewidth=2, label='Traditional')
ax.plot(t_dense, y_reg, 'b-', linewidth=2, label='Regressor', alpha=0.8)
ax.set_xlabel('Time t [s]')
ax.set_ylabel('Displacement y(t)')
ax.set_title(f'(a) Solution comparison (N={N})')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Absolute errors
ax = axes[0, 1]
err_trad = np.abs(y_trad - y_ref)
err_reg = np.abs(y_reg - y_ref)
ax.semilogy(t_dense, err_trad, 'r--', linewidth=2, label=f'Traditional (max={err_trad.max():.2e})')
ax.semilogy(t_dense, err_reg, 'b-', linewidth=2, label=f'Regressor (max={err_reg.max():.2e})')
ax.set_xlabel('Time t [s]')
ax.set_ylabel('Absolute error |y - y_ref|')
ax.set_title('(b) Error evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 3: f(y) approximation
ax = axes[1, 0]
y_eval = np.linspace(y_data.min(), y_data.max(), 200)
f_eval_true = true_spring_force(y_eval)
f_eval_trad = rbf_trad.eval(y_eval)
f_eval_reg = rbf_reg.eval(y_eval)
ax.plot(y_data, f_true, 'ko', markersize=8, label='Data', zorder=5)
ax.plot(y_eval, f_eval_true, 'k-', linewidth=2, label='True f(y)', alpha=0.7)
ax.plot(y_eval, f_eval_trad, 'r--', linewidth=2, label='Traditional')
ax.plot(y_eval, f_eval_reg, 'b-', linewidth=2, label='Regressor', alpha=0.8)
ax.set_xlabel('Displacement y')
ax.set_ylabel('Force f(y)')
ax.set_title('(c) Nonlinear force approximation')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Error table
ax = axes[1, 1]
ax.axis('off')
rmse_trad = np.sqrt(np.mean((y_trad - y_ref)**2))
rmse_reg = np.sqrt(np.mean((y_reg - y_ref)**2))
mejora = (rmse_trad - rmse_reg) / rmse_trad * 100

table_data = [
    ['Metric', 'Traditional', 'Regressor', 'Improvement'],
    ['RMSE y(t)', f'{rmse_trad:.3e}', f'{rmse_reg:.3e}', f'{mejora:+.1f}%'],
    ['Max Error', f'{err_trad.max():.3e}', f'{err_reg.max():.3e}', ''],
    ['Data points', str(N), str(N), ''],
    ['Step T', f'{T_MAX/(N-1):.3f} s', f'{T_MAX/(N-1):.3f} s', ''],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('(d) Performance metrics', pad=20)

plt.tight_layout()
plt.savefig('/home/rodo/1Paper/CaseStudy_4/duffing_regressor_comparison_N10.png',
            dpi=150, bbox_inches='tight')
print("✓ Figure saved: duffing_regressor_comparison_N10.png")

# ==============================================================================
# FIGURE 2: Sensitivity analysis (N vs Error)
# ==============================================================================
print("\n[2/3] Generating figure 2: Sensitivity analysis...")

N_values = [10, 15, 20, 25, 30, 40, 50]
errors_trad = []
errors_reg = []
times_trad = []
times_reg = []

for N in N_values:
    print(f"  Processing N={N}...")
    t_data = np.linspace(0, T_MAX, N)
    y_data = sol_ref.sol(t_data)[0]
    f_true = true_spring_force(y_data)

    # Traditional
    t0 = time.time()
    rbf_trad, _ = traditional_method(t_data, y_data, n_centers=max(3, N//5))
    y_trad = solve_duffing_regressor(rbf_trad, t_dense, y0, sol_ref.sol(t_data)[1, 0])
    times_trad.append(time.time() - t0)
    errors_trad.append(np.sqrt(np.mean((y_trad - y_ref)**2)))

    # Regressor
    t0 = time.time()
    rbf_reg = train_rbf(y_data, f_true, n_centers=max(3, N//5))
    y_reg = solve_duffing_regressor(rbf_reg, t_dense, y0, v0)
    times_reg.append(time.time() - t0)
    errors_reg.append(np.sqrt(np.mean((y_reg - y_ref)**2)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Subplot 1: Error vs N
ax = axes[0]
ax.semilogy(N_values, errors_trad, 'r-o', linewidth=2, markersize=8, label='Traditional')
ax.semilogy(N_values, errors_reg, 'b-s', linewidth=2, markersize=8, label='Regressor')
ax.axvline(20, color='gray', linestyle=':', alpha=0.5, label='Transition')
ax.set_xlabel('Number of points N')
ax.set_ylabel('RMSE of y(t)')
ax.set_title('(a) Error vs amount of data')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Relative improvement
ax = axes[1]
mejora_pct = [(errors_trad[i] - errors_reg[i])/errors_trad[i]*100
              for i in range(len(N_values))]
colors = ['green' if m > 0 else 'red' for m in mejora_pct]
ax.bar(N_values, mejora_pct, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.axvline(20, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of points N')
ax.set_ylabel('Regressor improvement (%)')
ax.set_title('(b) Relative improvement')
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Computation time
ax = axes[2]
ax.semilogy(N_values, times_trad, 'r-o', linewidth=2, markersize=8, label='Traditional')
ax.semilogy(N_values, times_reg, 'b-s', linewidth=2, markersize=8, label='Regressor')
ax.set_xlabel('Number of points N')
ax.set_ylabel('Computation time [s]')
ax.set_title('(c) Computational efficiency')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/rodo/1Paper/CaseStudy_4/duffing_sensitivity_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ Figure saved: duffing_sensitivity_analysis.png")

# ==============================================================================
# FIGURE 3: RBF approximation comparison for f(y)
# ==============================================================================
print("\n[3/3] Generating figure 3: RBF comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, (N, ax) in enumerate(zip([10, 20, 30, 50], axes.flatten())):
    t_data = np.linspace(0, T_MAX, N)
    y_data = sol_ref.sol(t_data)[0]
    f_true = true_spring_force(y_data)

    # Train RBFs
    rbf_trad, f_despeje = traditional_method(t_data, y_data, n_centers=max(3, N//5))
    rbf_reg = train_rbf(y_data, f_true, n_centers=max(3, N//5))

    # Evaluate
    y_eval = np.linspace(y_data.min()-0.2, y_data.max()+0.2, 300)
    f_eval_true = true_spring_force(y_eval)
    f_eval_trad = rbf_trad.eval(y_eval)
    f_eval_reg = rbf_reg.eval(y_eval)

    # Plot
    ax.plot(y_data, f_true, 'ko', markersize=6, label='Data', zorder=5)
    ax.plot(y_eval, f_eval_true, 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(y_eval, f_eval_trad, 'r--', linewidth=2, label='Traditional', alpha=0.8)
    ax.plot(y_eval, f_eval_reg, 'b-', linewidth=2, label='Regressor', alpha=0.8)

    # Calculate errors
    f_true_eval = true_spring_force(y_data)
    f_trad_eval = rbf_trad.eval(y_data)
    f_reg_eval = rbf_reg.eval(y_data)
    rmse_trad = np.sqrt(np.mean((f_trad_eval - f_true_eval)**2))
    rmse_reg = np.sqrt(np.mean((f_reg_eval - f_true_eval)**2))

    title = f'({chr(97+idx)}) N={N}, RMSE: Trad={rmse_trad:.2e}, Reg={rmse_reg:.2e}'
    ax.set_title(title)
    ax.set_xlabel('y')
    ax.set_ylabel('f(y)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/rodo/1Paper/CaseStudy_4/duffing_rbf_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ Figure saved: duffing_rbf_comparison.png")

print("\n" + "="*70)
print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)
