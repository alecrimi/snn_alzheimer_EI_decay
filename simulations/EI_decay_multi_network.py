# ============================================================
# PyNEST Multi-Subnetwork Simulation
# Runs 4 separate simulations (one per band)
# Stitches power spectra from corresponding frequency ranges
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import os


def load_connectivity_matrix(group, band, data_root="./"):
    """Load functional connectivity matrix from PLV data."""
    conn_file = os.path.join(data_root, group, f'connectivity_{band}', f'average_{band}_plv.npy')
    
    if not os.path.exists(conn_file):
        raise FileNotFoundError(f"Connectivity file not found: {conn_file}")
    
    conn_matrix = np.load(conn_file)
    np.fill_diagonal(conn_matrix, 0)
    
    print(f"    Loaded {band} FC: range [{conn_matrix.min():.3f}, {conn_matrix.max():.3f}]")
    
    return conn_matrix


def select_subnetworks(conn_matrix, n_subnets=15):
    """Select nodes with highest connectivity."""
    node_strength = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_subnets:]
    subnet_conn = conn_matrix[np.ix_(selected_indices, selected_indices)]
    
    return selected_indices, subnet_conn


def run_simulation_with_fc(condition_name,
                           band_name,
                           g_ratio,
                           conn_matrix,
                           N_subnet=200,
                           N_subnets=15,
                           frac_exc=0.8,
                           p_conn_local=0.15,
                           coupling_strength=2.0,
                           nu_ext=5.0,
                           sim_time=5000.0,
                           warmup=1000.0,
                           seed_base=42):
    """
    Run single simulation with specific FC matrix.
    """
    
    # Kernel setup
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1
    
    rng_seed = seed_base + hash(condition_name + band_name) % 1000
    np.random.seed(rng_seed)
    
    # Create subnetworks
    N_E_subnet = int(N_subnet * frac_exc)
    N_I_subnet = N_subnet - N_E_subnet
    
    E_populations = []
    I_populations = []
    
    for i in range(N_subnets):
        E_pop = nest.Create("iaf_cond_exp", N_E_subnet)
        I_pop = nest.Create("iaf_cond_exp", N_I_subnet)
        
        E_pop.V_m = -70.0 + 5.0 * np.random.randn(N_E_subnet)
        I_pop.V_m = -70.0 + 5.0 * np.random.randn(N_I_subnet)
        E_pop.V_th = -50.0 + 2.0 * np.random.randn(N_E_subnet)
        I_pop.V_th = -50.0 + 2.0 * np.random.randn(N_I_subnet)
        
        E_pop.t_ref = 2.0
        I_pop.t_ref = 1.0
        E_pop.tau_syn_ex = 2.0
        E_pop.tau_syn_in = 8.0
        I_pop.tau_syn_ex = 1.0
        I_pop.tau_syn_in = 4.0
        
        E_populations.append(E_pop)
        I_populations.append(I_pop)
    
    # External drive
    g_E_base = 3.0
    delay_local = 1.5
    delay_inter = 3.0
    
    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]
        
        for _ in range(N_subnet):
            pg = nest.Create("poisson_generator")
            rate = nu_ext * 1000.0 * np.random.lognormal(mean=0.0, sigma=0.2)
            pg.rate = rate
            
            if np.random.rand() < frac_exc:
                target = E_pop[np.random.randint(0, N_E_subnet)]
            else:
                target = I_pop[np.random.randint(0, N_I_subnet)]
            
            nest.Connect(pg, target, syn_spec={"weight": g_E_base, "delay": delay_local})
        
        noise = nest.Create("poisson_generator")
        noise.rate = 200.0
        nest.Connect(noise, E_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay_local})
        nest.Connect(noise, I_pop, syn_spec={"weight": 0.3 * g_E_base, "delay": delay_local})
    
    # Local connectivity
    g_E = g_E_base
    g_I = g_ratio * g_E
    
    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn_local}
    
    for subnet_idx in range(N_subnets):
        E_pop = E_populations[subnet_idx]
        I_pop = I_populations[subnet_idx]
        
        nest.Connect(E_pop, E_pop, conn_spec, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(E_pop, I_pop, conn_spec, syn_spec={"weight": 1.2 * g_E, "delay": delay_local})
        nest.Connect(I_pop, E_pop, conn_spec, syn_spec={"weight": -g_I, "delay": delay_local})
        nest.Connect(I_pop, I_pop, conn_spec, syn_spec={"weight": -0.8 * g_I, "delay": delay_local})
    
    # Inter-subnetwork connectivity based on THIS band's FC
    conn_normalized = conn_matrix / np.max(conn_matrix) if np.max(conn_matrix) > 0 else conn_matrix
    
    for i in range(N_subnets):
        for j in range(N_subnets):
            if i == j:
                continue
            
            fc_weight = conn_normalized[i, j]
            
            if fc_weight > 0.1:
                inter_weight = coupling_strength * g_E * fc_weight
                conn_spec_inter = {"rule": "pairwise_bernoulli", "p": 0.05}
                
                nest.Connect(E_populations[i], E_populations[j], 
                           conn_spec_inter,
                           syn_spec={"weight": inter_weight, "delay": delay_inter})
    
    # Recording
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"], interval=1.0)
    nest.Connect(mE, E_populations[0][:20])
    
    spike_recorders = []
    for subnet_idx in range(N_subnets):
        sr = nest.Create("spike_recorder")
        nest.Connect(E_populations[subnet_idx], sr)
        spike_recorders.append(sr)
    
    # Simulation
    nest.Simulate(warmup)
    nest.Simulate(sim_time)
    
    # Analysis
    spike_rates = []
    for sr in spike_recorders:
        spikes = sr.get("events")
        spike_times = spikes["times"]
        spike_times = spike_times[spike_times > warmup]
        rate = len(spike_times) / (N_E_subnet * sim_time / 1000.0)
        spike_rates.append(rate)
    
    # LFP proxy
    events = mE.get("events")
    if "times" not in events or len(events["times"]) < 2000:
        return {'success': False}
    
    t = np.array(events["times"])
    V_m = np.array(events["V_m"])
    
    mask = t > warmup
    V_m = V_m[mask]
    
    if len(V_m) < 1000:
        return {'success': False}
    
    fs = 1000.0 / mE.interval
    nperseg = min(4096, len(V_m) // 4)
    
    f, Pxx = welch(V_m, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)
    
    # Keep full spectrum (1-40 Hz)
    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    
    Pxx_rel = Pxx / np.sum(Pxx)
    
    return {
        'condition': condition_name,
        'band': band_name,
        'g_ratio': g_ratio,
        'f': f,
        'Pxx': Pxx_rel,
        'spike_rate': np.mean(spike_rates),
        'spike_rates_per_subnet': spike_rates,
        'success': True
    }


def stitch_spectra(results_by_condition, band_ranges):
    """
    Stitch together power spectra from different band simulations.
    
    Parameters:
    -----------
    results_by_condition : dict
        {condition: {band: result_dict}}
    band_ranges : dict
        {band: (low_freq, high_freq)}
    
    Returns:
    --------
    stitched_results : dict
        {condition: {'f': freq_array, 'Pxx': power_array}}
    """
    stitched = {}
    
    for condition, band_results in results_by_condition.items():
        # Initialize arrays for stitched spectrum
        all_f = []
        all_Pxx = []
        
        # Process each band in order
        for band_name in ['theta', 'alpha', 'beta', 'gamma']:
            if band_name not in band_results:
                print(f"    Warning: Missing {band_name} for {condition}")
                continue
            
            result = band_results[band_name]
            f = result['f']
            Pxx = result['Pxx']
            
            # Extract only the relevant frequency range for this band
            low, high = band_ranges[band_name]
            mask = (f >= low) & (f < high)
            
            all_f.append(f[mask])
            all_Pxx.append(Pxx[mask])
        
        # Concatenate all segments
        if all_f:
            stitched_f = np.concatenate(all_f)
            stitched_Pxx = np.concatenate(all_Pxx)
            
            # Renormalize
            stitched_Pxx = stitched_Pxx / np.sum(stitched_Pxx)
            
            stitched[condition] = {
                'f': stitched_f,
                'Pxx': stitched_Pxx
            }
    
    return stitched


# ============================================================
# Main execution
# ============================================================

DATA_ROOT = "./"
BAND_RANGES = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
N_SUBNETS = 15
N_SUBNET = 200

CONDITIONS = {
    'HC': 6.5,
    'AD': 2.5
}

print("="*70)
print("MULTI-SUBNETWORK SIMULATION - STITCHED SPECTRA")
print("="*70)
print(f"Running 4 separate simulations per condition (one per band)")
print(f"Then stitching frequency ranges:")
print(f"  Theta FC → 4-8 Hz")
print(f"  Alpha FC → 8-13 Hz")
print(f"  Beta FC → 13-30 Hz")
print(f"  Gamma FC → 30-45 Hz")
print("="*70)

# Store results: {condition: {band: result}}
results_by_condition = {condition: {} for condition in CONDITIONS.keys()}

# Run simulations
for condition, g_ratio in CONDITIONS.items():
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition} (g_I/g_E = {g_ratio})")
    print(f"{'='*70}")
    
    for band_name, (low, high) in BAND_RANGES.items():
        print(f"\n  [{band_name.upper()}] {low}-{high} Hz")
        
        try:
            # Load FC for this band
            conn_matrix = load_connectivity_matrix(condition, band_name, DATA_ROOT)
            
            # Select subnetworks
            selected_idx, subnet_conn = select_subnetworks(conn_matrix, N_SUBNETS)
            print(f"    Selected {N_SUBNETS} nodes: {selected_idx}")
            
            # Run simulation
            print(f"    Running simulation...")
            result = run_simulation_with_fc(
                condition_name=condition,
                band_name=band_name,
                g_ratio=g_ratio,
                conn_matrix=subnet_conn,
                N_subnet=N_SUBNET,
                N_subnets=N_SUBNETS,
                coupling_strength=2.0,
                sim_time=5000.0,
                warmup=1000.0
            )
            
            if result.get('success', False):
                results_by_condition[condition][band_name] = result
                print(f"    ✓ Success (spike rate: {result['spike_rate']:.2f} Hz)")
            else:
                print(f"    ✗ Failed")
        
        except Exception as e:
            print(f"    ✗ Error: {e}")

# ============================================================
# Stitch spectra
# ============================================================
print(f"\n{'='*70}")
print("STITCHING SPECTRA")
print(f"{'='*70}")

stitched_results = stitch_spectra(results_by_condition, BAND_RANGES)

for condition, data in stitched_results.items():
    print(f"{condition}: {len(data['f'])} frequency points, range [{data['f'].min():.1f}, {data['f'].max():.1f}] Hz")

# ============================================================
# Plotting
# ============================================================
if len(stitched_results) >= 2:
    print(f"\n{'='*70}")
    print("PLOTTING")
    print(f"{'='*70}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "HC": "#A9A9A9",
        "AD": "#90EE90"
    }
    
    # Plot stitched spectra
    for condition, data in stitched_results.items():
        g_ratio = CONDITIONS[condition]
        ax.plot(data['f'], data['Pxx'], 
               label=f"{condition} (g_ratio={g_ratio:.1f})",
               linewidth=2.5, 
               color=colors[condition], 
               alpha=0.85)
    
    # Add vertical lines at band boundaries
    band_boundaries = [4, 8, 13, 30]
    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add band labels
    y_max = ax.get_ylim()[1]
    band_centers = [(4+8)/2, (8+13)/2, (13+30)/2, (30+45)/2]
    band_names = ['Theta\n(Theta FC)', 'Alpha\n(Alpha FC)', 'Beta\n(Beta FC)', 'Gamma\n(Gamma FC)']
    
    for center, name in zip(band_centers, band_names):
        ax.text(center, y_max * 0.92, name, 
               horizontalalignment='center', 
               fontsize=9, 
               style='italic',
               color='red',
               alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
    
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative power", fontsize=14, fontweight='bold')
    ax.set_xlim([4, 45])
    ax.set_title("Stitched Power Spectrum\n(Each band uses its corresponding FC matrix)", 
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("stitched_spectrum.png", dpi=300, bbox_inches='tight')
    plt.savefig("stitched_spectrum.pdf", bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: stitched_spectrum.png/pdf")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
