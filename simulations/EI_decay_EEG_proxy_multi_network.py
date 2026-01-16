# ============================================================
# PyNEST – E/I imbalance with FC-based connectivity
# Conductance-based LIF neurons with smoothing
# Runs 4 separate simulations (one per band)
# Stitches power spectra from corresponding frequency ranges
# ============================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
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


def select_regions(conn_matrix, n_regions=19):
    """Select regions with highest connectivity."""
    node_strength = np.sum(np.abs(conn_matrix), axis=1)
    selected_indices = np.argsort(node_strength)[-n_regions:]
    region_conn = conn_matrix[np.ix_(selected_indices, selected_indices)]
    
    return selected_indices, region_conn


def run_simulation_with_fc(condition, 
                           band_name,
                           g_ratio,
                           conn_matrix,
                           N_per_region=100,
                           n_regions=19,
                           frac_exc=0.8,
                           p_conn_local=0.1,
                           coupling_strength=1.5,
                           nu_ext=8.0,
                           sim_time=6000.0,
                           warmup=2000.0,
                           seed=42,
                           record_fraction=0.15,
                           smooth_spectrum=True):
    """
    Run E/I network with conductance-based neurons and FC-weighted inter-region connections.
    
    Parameters
    ----------
    record_fraction : float
        Fraction of excitatory neurons to record from per region (default: 0.15)
    smooth_spectrum : bool
        Apply Savitzky-Golay smoothing to power spectrum (default: True)
    """

    # --------------------
    # NEST kernel setup
    # --------------------
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.set_verbosity("M_WARNING")
    
    # Use seed based on condition and band
    rng_seed = seed + hash(condition + band_name) % 1000
    np.random.seed(rng_seed)
    nest.SetKernelStatus({"rng_seed": rng_seed})

    print(f"\n[{condition} - {band_name}] g_I/g_E = {g_ratio:.2f}, seed = {rng_seed}")

    # --------------------
    # Create regional populations (CONDUCTANCE-BASED)
    # --------------------
    N_E_region = int(N_per_region * frac_exc)
    N_I_region = N_per_region - N_E_region
    
    # Conductance-based neuron parameters
    neuron_params = {
        "C_m": 250.0,          # pF - membrane capacitance
        "g_L": 16.67,          # nS - leak conductance
        "E_L": -70.0,          # mV - leak reversal potential
        "V_th": -50.0,         # mV - spike threshold
        "V_reset": -70.0,      # mV - reset potential
        "t_ref": 2.0,          # ms - refractory period
        "E_ex": 0.0,           # mV - excitatory reversal potential
        "E_in": -80.0,         # mV - inhibitory reversal potential
        "tau_syn_ex": 2.0,     # ms - excitatory synaptic time constant
        "tau_syn_in": 8.0,     # ms - inhibitory synaptic time constant
    }
    
    E_regions = []
    I_regions = []
    
    for i in range(n_regions):
        # Create conductance-based neurons
        E = nest.Create("iaf_cond_exp", N_E_region)
        I = nest.Create("iaf_cond_exp", N_I_region)
        
        # Set parameters
        E.set(neuron_params)
        I.set(neuron_params)
        
        # Initialize with variability
        E.V_m = -70.0 + 5.0 * np.random.randn(N_E_region)
        I.V_m = -70.0 + 5.0 * np.random.randn(N_I_region)
        
        E_regions.append(E)
        I_regions.append(I)

    # --------------------
    # Synaptic strengths (conductances in nS)
    # --------------------
    g_E = 2.0  # nS - excitatory synaptic conductance
    g_I = g_ratio * g_E  # nS - inhibitory synaptic conductance
    delay_local = 1.5
    delay_inter = 3.0

    # --------------------
    # External drive to each region
    # --------------------
    for region_idx in range(n_regions):
        ext = nest.Create("poisson_generator")
        ext.rate = nu_ext * 1000.0  # Hz
        
        nest.Connect(ext, E_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(ext, I_regions[region_idx], syn_spec={"weight": g_E, "delay": delay_local})

    # --------------------
    # Local recurrent connections within each region
    # --------------------
    conn = {"rule": "pairwise_bernoulli", "p": p_conn_local}
    
    for region_idx in range(n_regions):
        E = E_regions[region_idx]
        I = I_regions[region_idx]
        
        # Note: conductance-based uses positive weights, inhibition through E_in
        nest.Connect(E, E, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(E, I, conn, syn_spec={"weight": g_E, "delay": delay_local})
        nest.Connect(I, E, conn, syn_spec={"weight": g_I, "delay": delay_local})
        nest.Connect(I, I, conn, syn_spec={"weight": g_I, "delay": delay_local})

    # --------------------
    # Inter-region connections based on FC matrix
    # --------------------
    # Normalize FC matrix
    conn_normalized = conn_matrix / np.max(conn_matrix) if np.max(conn_matrix) > 0 else conn_matrix
    
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            
            fc_weight = conn_normalized[i, j]
            
            # Only connect if FC is above threshold
            if fc_weight > 0.1:
                # Weight inter-region connections by FC strength
                inter_weight = coupling_strength * g_E * fc_weight
                conn_inter = {"rule": "pairwise_bernoulli", "p": 0.05}
                
                # E->E connections between regions
                nest.Connect(E_regions[i], E_regions[j], 
                           conn_inter,
                           syn_spec={"weight": inter_weight, "delay": delay_inter})

    # --------------------
    # Multimeter - scale recording with network size
    # --------------------
    n_rec_per_region = max(10, int(record_fraction * N_E_region))
    n_rec_per_region = min(n_rec_per_region, N_E_region)
    
    mm = nest.Create("multimeter")
    mm.set({
        "interval": 1.0,
        "record_from": ["g_ex", "g_in", "V_m"]  # Conductance-based recordings
    })
    
    # Record from first region
    nest.Connect(mm, E_regions[0][:n_rec_per_region])
    print(f"    Recording from {n_rec_per_region}/{N_E_region} neurons per region ({100*n_rec_per_region/N_E_region:.1f}%)")

    # --------------------
    # Spike recorder
    # --------------------
    spike_rec = nest.Create("spike_recorder")
    for E in E_regions:
        nest.Connect(E, spike_rec)

    # --------------------
    # Simulate
    # --------------------
    nest.Simulate(warmup)
    nest.Simulate(sim_time)

    # --------------------
    # EEG / LFP proxy (from conductance-based recordings)
    # --------------------
    ev = mm.get("events")
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    g_ex = np.array(ev["g_ex"])  # nS
    g_in = np.array(ev["g_in"])  # nS
    V_m = np.array(ev["V_m"])    # mV
    
    # Calculate synaptic currents from conductances
    E_ex = 0.0   # mV
    E_in = -80.0 # mV
    
    I_ex = g_ex * (V_m - E_ex)  # pA
    I_in = g_in * (V_m - E_in)  # pA

    # Filter out warmup period
    mask = times > warmup
    times_filtered = times[mask]
    senders_filtered = senders[mask]
    I_ex_filtered = I_ex[mask]
    I_in_filtered = I_in[mask]

    # Check if we have enough data
    if len(times_filtered) < 1000:
        print(f"    WARNING: Insufficient data ({len(times_filtered)} points)")
        return {'success': False}

    # Get unique time points and neurons
    unique_times = np.sort(np.unique(times_filtered))
    unique_neurons = np.sort(np.unique(senders_filtered))
    n_times = len(unique_times)
    n_neurons = len(unique_neurons)
    
    print(f"    Recording from {n_neurons} neurons over {n_times} time points")

    # NEST records data chronologically
    expected_length = n_times * n_neurons
    
    if len(times_filtered) != expected_length:
        print(f"    WARNING: Expected {expected_length} points, got {len(times_filtered)}")
        # Handle missing data by explicit indexing
        I_ex_matrix = np.zeros((n_times, n_neurons))
        I_in_matrix = np.zeros((n_times, n_neurons))
        
        neuron_to_idx = {nid: idx for idx, nid in enumerate(unique_neurons)}
        time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
        
        for i in range(len(times_filtered)):
            t_idx = time_to_idx[times_filtered[i]]
            n_idx = neuron_to_idx[senders_filtered[i]]
            I_ex_matrix[t_idx, n_idx] = I_ex_filtered[i]
            I_in_matrix[t_idx, n_idx] = I_in_filtered[i]
    else:
        # Reshape directly
        I_ex_matrix = I_ex_filtered.reshape(n_times, n_neurons)
        I_in_matrix = I_in_filtered.reshape(n_times, n_neurons)

    # LFP proxy: average synaptic currents across neurons
    lfp = I_ex_matrix.mean(axis=1) - I_in_matrix.mean(axis=1)
    lfp -= lfp.mean()

    print(f"    LFP: {len(lfp)} samples, range [{lfp.min():.2f}, {lfp.max():.2f}] pA")

    # --------------------
    # Power spectrum with MAXIMUM smoothing
    # --------------------
    fs = 1000.0
    
    # Use very large window for smoothest spectra
    nperseg = min(len(lfp) // 2, 16384)
    nperseg = max(nperseg, 4096)
    
    # Maximum overlap (93.75%)
    noverlap = int(15 * nperseg // 16)

    f, Pxx = welch(lfp, fs=fs,
                   nperseg=nperseg,
                   noverlap=noverlap,
                   window='hann',
                   detrend='constant')

    # Keep full spectrum (1-40 Hz)
    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]
    
    # Apply Savitzky-Golay smoothing if requested
    if smooth_spectrum and len(Pxx) > 10:
        window_length = min(21, len(Pxx) // 2 * 2 - 1)  # Make it odd
        if window_length >= 5:
            Pxx = savgol_filter(Pxx, window_length=window_length, polyorder=3)
            print(f"    Applied Savitzky-Golay filter (window={window_length})")
    
    # Normalize
    Pxx = np.maximum(Pxx, 0)  # Ensure non-negative
    Pxx /= Pxx.sum()  # relative power

    # Calculate spike rate
    spikes = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]
    total_neurons = n_regions * N_E_region
    spike_rate = len(spike_times) / (total_neurons * sim_time / 1000.0)
    
    print(f"    Spike rate: {spike_rate:.2f} Hz, Welch: nperseg={nperseg}")

    return {
        "condition": condition,
        "band": band_name,
        "g_ratio": g_ratio,
        "f": f,
        "Pxx": Pxx,
        "lfp": lfp,
        "spike_rate": spike_rate,
        "success": True
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
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
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
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}
N_REGIONS = 19
N_PER_REGION = 40

CONDITIONS = {
    'HC': 6.5,
    'AD': 2.5
}

print("="*70)
print("E/I BALANCE WITH FC-BASED CONNECTIVITY - CONDUCTANCE-BASED NEURONS")
print("="*70)
print(f"Using conductance-based LIF neurons (iaf_cond_exp)")
print(f"Running 5 separate simulations per condition (one per band)")
print(f"With maximum spectral smoothing")
print(f"Then stitching frequency ranges:")
print(f"  Delta FC → 1-4 Hz")
print(f"  Theta FC → 4-8 Hz")
print(f"  Alpha FC → 8-13 Hz")
print(f"  Beta FC → 13-30 Hz")
print(f"  Gamma FC → 30-40 Hz")
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
            
            # Select regions
            selected_idx, region_conn = select_regions(conn_matrix, N_REGIONS)
            print(f"    Selected {N_REGIONS} regions: {selected_idx[:5]}...{selected_idx[-5:]}")
            
            # Run simulation with conductance-based neurons
            print(f"    Running simulation...")
            result = run_simulation_with_fc(
                condition=condition,
                band_name=band_name,
                g_ratio=g_ratio,
                conn_matrix=region_conn,
                N_per_region=N_PER_REGION,
                n_regions=N_REGIONS,
                coupling_strength=1.5,
                sim_time=6000.0,
                warmup=2000.0,
                seed=42,
                record_fraction=0.15,  # 15% of neurons per region
                smooth_spectrum=True    # Enable Savitzky-Golay smoothing
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
# Verification: Check that spectra are different
# ============================================================
print(f"\n{'='*70}")
print("SPECTRUM VERIFICATION")
print(f"{'='*70}")

conditions_list = list(stitched_results.keys())
if len(conditions_list) >= 2:
    for i in range(len(conditions_list)):
        for j in range(i+1, len(conditions_list)):
            cond1 = conditions_list[i]
            cond2 = conditions_list[j]
            Pxx1 = stitched_results[cond1]['Pxx']
            Pxx2 = stitched_results[cond2]['Pxx']
            
            # Ensure same length for correlation
            min_len = min(len(Pxx1), len(Pxx2))
            corr = np.corrcoef(Pxx1[:min_len], Pxx2[:min_len])[0, 1]
            print(f"✓ {cond1} vs {cond2}: correlation = {corr:.3f}")

# ============================================================
# Plotting
# ============================================================
if len(stitched_results) >= 2:
    print(f"\n{'='*70}")
    print("PLOTTING")
    print(f"{'='*70}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "AD": "#90EE90",
        "HC": "#A9A9A9",
    }
    
    # Plot stitched spectra
    for condition in ['AD', 'HC']:
        if condition in stitched_results:
            data = stitched_results[condition]
            g_ratio = CONDITIONS[condition]
            ax.plot(data['f'], data['Pxx'], 
                   label=f"{condition}",
                   linewidth=2.5, 
                   color=colors[condition], 
                   alpha=0.85)
    
    # Add vertical dashed lines for frequency band boundaries
    band_boundaries = [4, 8, 13, 30]
    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Add band labels at the top
    y_max = ax.get_ylim()[1]
    band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for center, name in zip(band_centers, band_names):
        ax.text(center, y_max * 0.95, name, 
               horizontalalignment='center', 
               fontsize=10, 
               style='italic',
               color='gray',
               alpha=0.7)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative power", fontsize=12)
    ax.set_xlim([1, 40])
    ax.set_title("E/I Balance with FC-Based Connectivity (Conductance-based LIF)\n(Each band uses its corresponding FC matrix)", 
                fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("EI_FC_stitched_spectrum_conductance_smooth.png", dpi=300)
    print("✓ Saved: EI_FC_stitched_spectrum_conductance_smooth.png")
    plt.close()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
