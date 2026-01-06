# ============================================================
# PyNEST prototype  
# E/I imbalance model  
# g_ratio = g_I / g_E  (higher -> stronger inhibition)
# ============================================================

import nest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


def run_simulation(condition_name,
                   g_ratio,
                   N_total=200,
                   frac_exc=0.8,
                   p_conn=0.15,
                   nu_ext=5.0,
                   sim_time=5000.0,
                   warmup=1000.0,
                   seed_base=42):
    """Run an E/I balanced network simulation with robust analysis."""

    # --------------------
    # Kernel setup
    # --------------------
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.resolution = 0.1  # ms

    rng_seed = seed_base + int(10 * g_ratio)
    np.random.seed(rng_seed)

    # --------------------
    # Populations
    # --------------------
    N_E = int(N_total * frac_exc)
    N_I = N_total - N_E

    E_pop = nest.Create("iaf_cond_exp", N_E)
    I_pop = nest.Create("iaf_cond_exp", N_I)

    # --------------------
    # Neuronal heterogeneity
    # --------------------
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

    # --------------------
    # External Poisson drive (robust, non-negative)
    # --------------------
    g_E_base = 3.0  # reduced for stability
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

    # --------------------
    # Recurrent connectivity
    # --------------------
    g_E = g_E_base
    g_I = g_ratio * g_E

    assert np.isclose(g_I / g_E, g_ratio)

    print(f"    g_E = {g_E:.2f}, g_I = {g_I:.2f} (ratio = {g_ratio:.2f})")

    conn_spec = {"rule": "pairwise_bernoulli", "p": p_conn}

    nest.Connect(E_pop, E_pop, conn_spec, syn_spec={"weight": g_E, "delay": delay})
    nest.Connect(E_pop, I_pop, conn_spec, syn_spec={"weight": 1.2 * g_E, "delay": delay})
    nest.Connect(I_pop, E_pop, conn_spec, syn_spec={"weight": -g_I, "delay": delay})
    nest.Connect(I_pop, I_pop, conn_spec, syn_spec={"weight": -0.8 * g_I, "delay": delay})

    # --------------------
    # Recording
    # --------------------
    mE = nest.Create("multimeter")
    mE.set(record_from=["V_m"], interval=1.0)
    nest.Connect(mE, E_pop[:20])

    spike_rec = nest.Create("spike_recorder")
    nest.Connect(E_pop, spike_rec)

    # --------------------
    # Simulation
    # --------------------
    print(f"    Warming up: {warmup} ms")
    nest.Simulate(warmup)

    print(f"    Recording: {sim_time} ms")
    nest.Simulate(sim_time)

    # --------------------
    # Spike-rate analysis (post-warmup)
    # --------------------
    spikes = spike_rec.get("events")
    spike_times = spikes["times"]
    spike_times = spike_times[spike_times > warmup]

    spike_rate = len(spike_times) / (N_E * sim_time / 1000.0)
    print(f"    Spike rate: {spike_rate:.2f} Hz/neuron")

    # --------------------
    # LFP proxy (single neuron, robust)
    # --------------------
    events = mE.get("events")
    if "times" not in events or len(events["times"]) < 2000:
        print("    ✗ Insufficient membrane potential data")
        return {'success': False}

    t = np.array(events["times"])
    V_m = np.array(events["V_m"])

    mask = t > warmup
    V_m = V_m[mask]

    if len(V_m) < 1000:
        print("    ✗ Too little post-warmup data")
        return {'success': False}

    fs = 1000.0 / mE.interval
    nperseg = min(4096, len(V_m) // 4)

    f, Pxx = welch(V_m, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)

    band = (f >= 1) & (f <= 40)
    f = f[band]
    Pxx = Pxx[band]

    Pxx_rel = Pxx / np.sum(Pxx)

    print("    ✓ Spectrum computed")

    return {
        'condition': condition_name,
        'g_ratio': g_ratio,
        'f': f,
        'Pxx': Pxx_rel,
        'spike_rate': spike_rate,
        'success': True
    }


# ============================================================
# Main execution
# ============================================================

conditions = [
    ("AD", 2.5), #"Low inhibition"
    ("MCI", 3.5), # "Medium inhibition"
    ("HC", 6.5) #, # High inhibition
#    ("HC, Very high inhibition", 8.5),
]

all_spectra = []

print("Starting simulations")
print("=" * 60)

for name, g_ratio in conditions:
    print(f"\n[{name}] g_I/g_E = {g_ratio}")
    res = run_simulation(name, g_ratio)
    if res.get('success', False):
        all_spectra.append(res)

print("\n" + "=" * 60)
print(f"Successful: {len(all_spectra)}/{len(conditions)}")

# --------------------
# Plotting
# --------------------
if all_spectra:
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "AD": "#90EE90",
        "MCI": "#FFD700",
        "HC": "#A9A9A9",
      #  "Very high inhibition": "#FF6347",
    }

    for s in all_spectra:
        ax.plot(s['f'], s['Pxx'], label=s['condition'],
                linewidth=2.5, color=colors[s['condition']], alpha=0.85)

    # Add vertical dashed lines for frequency band boundaries
    # Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz, Gamma: 30-45 Hz
    band_boundaries = [4, 8, 13, 30]
    
    for boundary in band_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Add band labels at the top
    y_max = ax.get_ylim()[1]
    band_centers = [(1+4)/2, (4+8)/2, (8+13)/2, (13+30)/2, (30+40)/2]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for center, name in zip(band_centers, band_names):
        if center <= 40:  # Only show labels within x-axis range
            ax.text(center, y_max * 0.95, name, 
                   horizontalalignment='center', 
                   fontsize=10, 
                   style='italic',
                   color='gray',
                   alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative power", fontsize=14, fontweight='bold')
    ax.set_xlim([1, 40])
    ax.grid(alpha=0.2)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("power_spectra_EI_fixed.png", dpi=300)
    plt.savefig("power_spectra_EI_fixed.pdf")

print("Done.")
