# ============================================================
# Enhanced PyNEST E/I Simulation with Full Randomization Control
# Multiple runs with different random network connectivity
# ============================================================

import nest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF

# ============================================================
# Cohen's d Functions
# ============================================================

def compute_cohens_d(group1_values, group2_values):
    """Compute Cohen's d with pooled standard deviation."""
    g1 = np.array(group1_values)
    g2 = np.array(group2_values)
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = np.mean(g1), np.mean(g2)
    std1 = np.std(g1, ddof=1)
    std2 = np.std(g2, ddof=1)
    
    s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / s_pooled
    
    return {
        'cohens_d': cohens_d,
        'mean_diff': mean1 - mean2,
        's_pooled': s_pooled,
        'mean_AD': mean1,
        'mean_HC': mean2,
        'std_AD': std1,
        'std_HC': std2,
        'n_AD': n1,
        'n_HC': n2
    }


# ============================================================
# NEST Simulation with Full Randomization Control
# ============================================================

def run_simulation(condition_name,
                   g_ratio,
                   N_total=200,
                   frac_exc=0.8,
                   p_conn=0.15,
                   nu_ext=5.0,
                   sim_time=5000.0,
                   warmup=1000.0,
                   seed_base=42):
    """
    Run E/I network simulation with full control over randomness.
    
    Sources of randomness controlled by seed_base:
    1. NEST RNG seeds (for stochastic connections)
    2. NumPy RNG (for initial conditions, parameters)
    3. Poisson generators (spike trains)
    
    Each simulation run with different seed_base will have:
    - Different random connectivity patterns
    - Different initial membrane potentials
    - Different threshold values  
    - Different external drive rates
    - Different spike timing
    """

    # ============================================================
    # CRITICAL: Set both NEST and NumPy RNG seeds
    # ============================================================
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1
    
    # Set NEST RNG seeds (controls stochastic connections!)
    # NEST uses multiple RNG threads - seed them all
    nest.rng_seed = seed_base
    nest.total_num_virtual_procs = 1  # Use single thread for reproducibility
    
    # Set NumPy RNG seed (controls parameter generation)
    np.random.seed(seed_base)

    # Populations
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E_pop = nest.Create("iaf_cond_exp", N_E)
    I_pop = nest.Create("iaf_cond_exp", N_I)

    # ============================================================
    # Neuronal heterogeneity (NumPy random - varies with seed)
    # ============================================================
    E_pop.V_m = -70.0 + 5.0 * np.random.randn(N_E)
    I_pop.V_m = -70.0 + 5.0 * np.random.randn(N_I)
    E_pop.V_th = -50.0 + 2.0 * np.random.randn(N_E)
    I_pop.V_th = -50.0 + 2.0 * np.random.randn(N_I)
    E_pop.t_ref = 2.0
    I_pop.t_ref = 1.0
    E_pop.tau_syn_ex = 2.0
    E_pop.tau_syn_in = 8.0
    I_pop.tau_syn_ex = 1.0
    I_pop.tau_syn_in = 4.0

    # ============================================================
    # External drive (rates vary with seed)
    # ============================================================
    g_E_base = 3.0
    delay = 1.5

    ext_drives = []
    for _ in range(N_total):
        pg = nest.Create("poisson_generator")
        rate = nu_ext * 1000.0 * np.random.lognormal(mean=0.0, sigma=0.2)
        pg.rate = rate
        ext_drives.append(pg)

    for i, pg in enumerate(ext_drives):
        if i < N_E:
            nest.Connect(pg, E_pop[i:i+1], syn_spec={"weight": g_E_base, "delay": delay})
        else:
            nest.Connect(pg, I_pop[i-N_E:i-N_E+1], syn_spec={"weight": g_E_base, "delay": delay})

    # Background noise
    noise = nest.Create("poisson_generator")
    noise.rate = 200.0
    nest.Connect(noise, E_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay})
    nest.Connect(noise, I_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay})

    # ============================================================
    # Recurrent connectivity (NEST RNG - varies with seed!)
    # This is where random connectivity patterns are generated
    # ============================================================
    g_E = g_E_base
    g_I = g_ratio * g_E

    # pairwise_bernoulli creates random connections based on NEST's RNG
    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn}
    
    nest.Connect(E_pop, E_pop, conn_spec, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E_pop, I_pop, conn_spec, syn_spec={"weight": 1.2 * g_E, "delay": delay})
    nest.Connect(I_pop, E_pop, conn_spec, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I_pop, I_pop, conn_spec, syn_spec={"weight": -0.8 * g_I, "delay": delay})

    # Recording
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"], interval=1.0)
    nest.Connect(mE, E_pop[:20])

    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E_pop, spike_rec)

    # Simulate
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # ============================================================
    # Analysis
    # ============================================================
    # Spike analysis
    spikes = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]
    spike_rate = len(spike_times) / (N_E * sim_time / 1000.0)

    # LFP proxy
    events = mE.get("events")
    t = np.array(events["times"])
    V_m = np.array(events["V_m"])
    mask = t > warmup
    V_m = V_m[mask]

    if len(V_m) < 1000:
        return None

    # Power spectrum
    fs = 1000.0 / mE.interval
    nperseg = min(4096, len(V_m) // 4)
    f, Pxx = welch(V_m, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)

    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]

    # Band powers
    def band_power(f, Pxx, fmin, fmax):
        idx = (f >= fmin) & (f < fmax)
        return np.mean(Pxx[idx]) if np.any(idx) else 0.0

    theta_power = band_power(f, Pxx, 4, 8)
    alpha_power = band_power(f, Pxx, 8, 13)
    beta_power = band_power(f, Pxx, 13, 30)
    gamma_power = band_power(f, Pxx, 30, 40)

    # Aperiodic fitting
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=5, min_peak_height=0.1)
    try:
        fm.fit(f, Pxx, freq_range=[1, 40])
        aperiodic_params = fm.get_params('aperiodic_params')
        offset = aperiodic_params[0]
        exponent = aperiodic_params[1]
    except:
        offset = np.nan
        exponent = np.nan

    return {
        'condition': condition_name,
        'g_ratio': g_ratio,
        'seed': seed_base,
        'aperiodic_exponent': exponent,
        'aperiodic_offset': offset,
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'spike_rate': spike_rate
    }


# ============================================================
# Multiple Simulations with Different Seeds
# ============================================================

def run_multiple_simulations(n_runs=10):
    """
    Run multiple simulations with different random seeds.
    
    Each run will have:
    - Different random connectivity patterns
    - Different initial conditions
    - Different external input patterns
    """
    
    print("="*70)
    print("RUNNING MULTIPLE SIMULATIONS WITH DIFFERENT RANDOM NETWORKS")
    print("="*70)
    print(f"Each run uses a different random seed to generate:")
    print("  • Random network connectivity (Bernoulli connections)")
    print("  • Random initial conditions (V_m, V_th)")
    print("  • Random external drive rates")
    print("="*70)
    
    ad_results = []
    hc_results = []
    
    for run in range(n_runs):
        # Use widely spaced seeds to ensure independence
        seed = 1000 + run * 1000
        
        print(f"\n{'='*70}")
        print(f"Run {run+1}/{n_runs} (seed={seed})")
        print(f"{'='*70}")
        
        # AD simulation
        print(f"  [AD] Running with g_ratio=2.5, seed={seed}...")
        res_ad = run_simulation("AD", 2.5, seed_base=seed)
        if res_ad:
            ad_results.append(res_ad)
            print(f"       ✓ Exponent: {res_ad['aperiodic_exponent']:.3f}, "
                  f"Spike rate: {res_ad['spike_rate']:.2f} Hz")
        else:
            print(f"       ✗ Failed")
        
        # HC simulation (different seed for truly independent network)
        seed_hc = seed + 100  # Offset to ensure different connectivity
        print(f"  [HC] Running with g_ratio=6.5, seed={seed_hc}...")
        res_hc = run_simulation("HC", 6.5, seed_base=seed_hc)
        if res_hc:
            hc_results.append(res_hc)
            print(f"       ✓ Exponent: {res_hc['aperiodic_exponent']:.3f}, "
                  f"Spike rate: {res_hc['spike_rate']:.2f} Hz")
        else:
            print(f"       ✗ Failed")
    
    print(f"\n{'='*70}")
    print(f"✓ Completed {len(ad_results)} AD and {len(hc_results)} HC simulations")
    print(f"{'='*70}")
    
    return ad_results, hc_results


def extract_measures(results_list):
    """Extract all measures into dictionary format."""
    measures = {}
    
    for key in ['aperiodic_exponent', 'aperiodic_offset', 'theta_power', 
                'alpha_power', 'beta_power', 'gamma_power', 'spike_rate']:
        measures[key] = [r[key] for r in results_list if not np.isnan(r[key])]
    
    return measures


def compute_all_cohens_d(ad_measures, hc_measures):
    """Compute Cohen's d for all measures."""
    results = []
    
    for measure in ad_measures.keys():
        if measure in hc_measures:
            result = compute_cohens_d(ad_measures[measure], hc_measures[measure])
            result['measure'] = measure
            results.append(result)
    
    df = pd.DataFrame(results)
    return df


def plot_cohens_d_comparison(sim_effects, emp_effects, output_prefix='cohens_d'):
    """Create comprehensive comparison plots."""
    
    comparison = pd.merge(
        sim_effects[['measure', 'cohens_d']], 
        emp_effects[['measure', 'cohens_d']], 
        on='measure', 
        suffixes=('_sim', '_emp')
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Bar comparison
    x_pos = np.arange(len(comparison))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, comparison['cohens_d_sim'], width, 
                label='Simulation', color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].bar(x_pos + width/2, comparison['cohens_d_emp'], width, 
                label='Empirical', color='#E27D60', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel("Cohen's d (AD - HC)", fontsize=13, fontweight='bold')
    axes[0].set_title('Effect Sizes: Simulation vs Empirical', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison['measure'], rotation=45, ha='right')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Correlation
    axes[1].scatter(comparison['cohens_d_emp'], comparison['cohens_d_sim'], 
                   s=150, color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=2)
    
    all_d = list(comparison['cohens_d_emp']) + list(comparison['cohens_d_sim'])
    lims = [min(all_d) - 0.3, max(all_d) + 0.3]
    axes[1].plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect match')
    
    if len(comparison) > 1:
        corr = np.corrcoef(comparison['cohens_d_emp'], comparison['cohens_d_sim'])[0, 1]
        axes[1].text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=axes[1].transAxes, fontsize=13, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    for idx, row in comparison.iterrows():
        axes[1].annotate(row['measure'], 
                        (row['cohens_d_emp'], row['cohens_d_sim']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    axes[1].set_xlabel("Empirical Cohen's d", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Simulation Cohen's d", fontsize=13, fontweight='bold')
    axes[1].set_title('Correlation Analysis', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Plot 3: Differences
    comparison['difference'] = comparison['cohens_d_sim'] - comparison['cohens_d_emp']
    colors = ['#2ECC71' if d > 0 else '#E74C3C' for d in comparison['difference']]
    
    axes[2].barh(x_pos, comparison['difference'], color=colors, 
                alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    axes[2].set_yticks(x_pos)
    axes[2].set_yticklabels(comparison['measure'])
    axes[2].set_xlabel("Δ Cohen's d (Sim - Emp)", fontsize=13, fontweight='bold')
    axes[2].set_title('Model Bias', fontsize=13, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_comparison.png/pdf")


def plot_variability_across_runs(ad_results, hc_results):
    """Visualize variability across different random network realizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    measures = ['aperiodic_exponent', 'theta_power', 'alpha_power', 
                'beta_power', 'gamma_power', 'spike_rate']
    
    for idx, measure in enumerate(measures):
        ad_vals = [r[measure] for r in ad_results if not np.isnan(r[measure])]
        hc_vals = [r[measure] for r in hc_results if not np.isnan(r[measure])]
        
        # Box plot
        bp = axes[idx].boxplot([ad_vals, hc_vals], labels=['AD', 'HC'],
                               patch_artist=True, widths=0.6)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('#90EE90')
        bp['boxes'][1].set_facecolor('#A9A9A9')
        
        # Add individual points
        for i, vals in enumerate([ad_vals, hc_vals], 1):
            x = np.random.normal(i, 0.04, len(vals))
            axes[idx].scatter(x, vals, alpha=0.5, s=30, c='black', zorder=3)
        
        axes[idx].set_ylabel(measure.replace('_', ' ').title(), fontsize=11)
        axes[idx].set_title(f'{measure.replace("_", " ").title()}\n'
                           f'(n_runs={len(ad_vals)})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
    
    plt.suptitle('Variability Across Different Random Network Realizations', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('network_variability.png', dpi=300, bbox_inches='tight')
    plt.savefig('network_variability.pdf', bbox_inches='tight')
    print("✓ Saved network_variability.png/pdf")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    # Run simulations with different random networks
    ad_results, hc_results = run_multiple_simulations(n_runs=10)
    
    # Extract measures
    ad_sim = extract_measures(ad_results)
    hc_sim = extract_measures(hc_results)
    
    # Compute Cohen's d
    sim_effects = compute_all_cohens_d(ad_sim, hc_sim)
    
    print("\n" + "="*70)
    print("SIMULATION EFFECT SIZES (AD vs HC)")
    print("="*70)
    print(sim_effects[['measure', 'cohens_d', 's_pooled', 'mean_AD', 'mean_HC']].to_string(index=False))
    
    # Plot variability
    plot_variability_across_runs(ad_results, hc_results)
    
    # Load empirical data (REPLACE WITH YOUR DATA)
    print("\n" + "="*70)
    print("EMPIRICAL DATA (PLACEHOLDER - REPLACE WITH YOUR DATA)")
    print("="*70)
    
    ad_empirical = {
        'aperiodic_exponent': np.random.normal(1.25, 0.18, 15),
        'aperiodic_offset': np.random.normal(-2.3, 0.35, 15),
        'theta_power': np.random.normal(0.24, 0.05, 15),
        'alpha_power': np.random.normal(0.21, 0.04, 15),
        'beta_power': np.random.normal(0.16, 0.03, 15),
        'gamma_power': np.random.normal(0.12, 0.02, 15),
        'spike_rate': np.random.normal(8.5, 1.2, 15)
    }
    
    hc_empirical = {
        'aperiodic_exponent': np.random.normal(1.75, 0.22, 20),
        'aperiodic_offset': np.random.normal(-2.9, 0.40, 20),
        'theta_power': np.random.normal(0.18, 0.04, 20),
        'alpha_power': np.random.normal(0.23, 0.05, 20),
        'beta_power': np.random.normal(0.19, 0.04, 20),
        'gamma_power': np.random.normal(0.14, 0.03, 20),
        'spike_rate': np.random.normal(7.2, 1.0, 20)
    }
    
    # Compute empirical Cohen's d
    emp_effects = compute_all_cohens_d(ad_empirical, hc_empirical)
    
    print("\n" + "="*70)
    print("EMPIRICAL EFFECT SIZES (AD vs HC)")
    print("="*70)
    print(emp_effects[['measure', 'cohens_d', 's_pooled', 'mean_AD', 'mean_HC']].to_string(index=False))
    
    # Compare
    plot_cohens_d_comparison(sim_effects, emp_effects)
    
    # Detailed comparison
    comparison = pd.merge(
        sim_effects[['measure', 'cohens_d', 's_pooled']], 
        emp_effects[['measure', 'cohens_d', 's_pooled']], 
        on='measure', 
        suffixes=('_sim', '_emp')
    )
    
    comparison['d_difference'] = comparison['cohens_d_sim'] - comparison['cohens_d_emp']
    comparison['d_ratio'] = comparison['cohens_d_sim'] / comparison['cohens_d_emp']
    
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Goodness of fit
    if len(comparison) > 1:
        corr = np.corrcoef(comparison['cohens_d_sim'], comparison['cohens_d_emp'])[0, 1]
        mae = np.mean(np.abs(comparison['d_difference']))
        rmse = np.sqrt(np.mean(comparison['d_difference']**2))
        
        print("\n" + "="*70)
        print("MODEL FIT METRICS")
        print("="*70)
        print(f"Pearson correlation (r): {corr:7.3f}")
        print(f"Mean Absolute Error:     {mae:7.3f}")
        print(f"Root Mean Square Error:  {rmse:7.3f}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  • cohens_d_comparison.png/pdf - Effect size comparison")
    print("  • network_variability.png/pdf - Variability across random networks")
