# ============================================================
# EEG 1/f Slope Analysis with Cohen's d
# Analyzes .set files from HC and AD folders
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF
import mne


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
    
    if s_pooled == 0:
        cohens_d = 0.0
    else:
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
# EEG File Discovery
# ============================================================

def find_eeg_files(base_dir, group_name):
    """
    Find all .set files in the directory structure:
    base_dir/sub-XXX/eeg/*.set
    
    Parameters
    ----------
    base_dir : str
        Path to HC or AD folder
    group_name : str
        Name of the group ('HC' or 'AD')
        
    Returns
    -------
    list of dict
        List of dictionaries with file paths and subject IDs
    """
    eeg_files = []
    
    # Search pattern: base_dir/sub-*/eeg/*.set
    pattern = os.path.join(base_dir, "sub-*", "eeg", "*.set")
    found_files = glob.glob(pattern)
    
    for filepath in found_files:
        # Extract subject ID from path
        path_parts = filepath.split(os.sep)
        
        # Find the sub-XXX part
        sub_id = None
        for part in path_parts:
            if part.startswith("sub-"):
                sub_id = part
                break
        
        if sub_id:
            eeg_files.append({
                'filepath': filepath,
                'subject_id': sub_id,
                'group': group_name,
                'filename': os.path.basename(filepath)
            })
    
    return eeg_files


# ============================================================
# EEG Processing and 1/f Slope Extraction
# ============================================================

def compute_1f_slope(raw, freq_range=[1, 40], channels='all'):
    """
    Compute 1/f slope (aperiodic exponent) from EEG data using FOOOF.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    freq_range : list
        Frequency range for FOOOF fitting [min, max] in Hz
    channels : str or list
        Which channels to use ('all' or list of channel names)
        
    Returns
    -------
    dict
        Dictionary with aperiodic parameters and band powers
    """
    
    # Select channels
    if channels == 'all':
        data = raw.get_data()
    else:
        picks = mne.pick_channels(raw.ch_names, include=channels)
        data = raw.get_data(picks=picks)
    
    # Average across channels
    data_avg = data.mean(axis=0)
    
    # Get sampling frequency
    sfreq = raw.info['sfreq']
    
    # Compute power spectrum using Welch's method
    nperseg = min(int(4 * sfreq), len(data_avg) // 4)  # 4 second windows
    nperseg = max(nperseg, 256)  # At least 256 samples
    
    noverlap = int(0.75 * nperseg)  # 75% overlap
    
    freqs, psd = welch(data_avg, fs=sfreq, nperseg=nperseg, 
                       noverlap=noverlap, window='hann')
    
    # Filter to frequency range of interest
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[freq_mask]
    psd_fit = psd[freq_mask]
    
    # Fit FOOOF model
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1,
               aperiodic_mode='fixed')  # Use 'fixed' for 1/f^exponent
    
    try:
        fm.fit(freqs_fit, psd_fit, freq_range=freq_range)
        
        # Extract aperiodic parameters
        aperiodic_params = fm.get_params('aperiodic_params')
        offset = aperiodic_params[0]
        exponent = aperiodic_params[1]
        
        # Extract periodic parameters
        n_peaks = len(fm.peak_params_)
        
        # Compute band powers from original PSD
        def band_power(freqs, psd, fmin, fmax):
            idx = (freqs >= fmin) & (freqs < fmax)
            return np.mean(psd[idx]) if np.any(idx) else np.nan
        
        theta_power = band_power(freqs, psd, 4, 8)
        alpha_power = band_power(freqs, psd, 8, 13)
        beta_power = band_power(freqs, psd, 13, 30)
        gamma_power = band_power(freqs, psd, 30, 40)
        
        # R-squared of the fit
        r_squared = fm.r_squared_
        
        return {
            'aperiodic_exponent': exponent,
            'aperiodic_offset': offset,
            'n_peaks': n_peaks,
            'r_squared': r_squared,
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'gamma_power': gamma_power,
            'success': True,
            'freqs': freqs_fit,
            'psd': psd_fit,
            'fooof_model': fm
        }
        
    except Exception as e:
        print(f"      ✗ FOOOF fitting failed: {str(e)}")
        return {
            'aperiodic_exponent': np.nan,
            'aperiodic_offset': np.nan,
            'n_peaks': np.nan,
            'r_squared': np.nan,
            'theta_power': np.nan,
            'alpha_power': np.nan,
            'beta_power': np.nan,
            'gamma_power': np.nan,
            'success': False
        }


def process_eeg_file(file_info, freq_range=[1, 40]):
    """
    Load and process a single EEG .set file.
    
    Parameters
    ----------
    file_info : dict
        Dictionary with filepath and subject info
    freq_range : list
        Frequency range for analysis
        
    Returns
    -------
    dict
        Results dictionary with subject info and computed metrics
    """
    
    try:
        # Load .set file using MNE
        raw = mne.io.read_raw_eeglab(file_info['filepath'], preload=True, verbose=False)
        
        '''
        # Basic preprocessing
        # 1. Apply band-pass filter
        raw.filter(l_freq=0.5, h_freq=45, fir_design='firwin', verbose=False)
        
        # 2. Remove bad channels if any
        if len(raw.info['bads']) > 0:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        
        # 3. Re-reference to average (optional, comment out if not needed)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        '''
        # Compute 1/f slope and other metrics
        results = compute_1f_slope(raw, freq_range=freq_range)
        
        # Add subject information
        results['subject_id'] = file_info['subject_id']
        results['group'] = file_info['group']
        results['filepath'] = file_info['filepath']
        results['filename'] = file_info['filename']
        results['n_channels'] = len(raw.ch_names)
        results['sfreq'] = raw.info['sfreq']
        results['duration'] = raw.times[-1]
        
        return results
        
    except Exception as e:
        print(f"      ✗ Error loading file: {str(e)}")
        return {
            'subject_id': file_info['subject_id'],
            'group': file_info['group'],
            'filepath': file_info['filepath'],
            'filename': file_info['filename'],
            'aperiodic_exponent': np.nan,
            'aperiodic_offset': np.nan,
            'success': False,
            'error': str(e)
        }


# ============================================================
# Main Analysis Pipeline
# ============================================================

def analyze_eeg_folders(hc_dir, ad_dir, freq_range=[1, 40]):
    """
    Main function to analyze EEG files from HC and AD folders.
    
    Parameters
    ----------
    hc_dir : str
        Path to HC folder
    ad_dir : str
        Path to AD folder
    freq_range : list
        Frequency range for FOOOF fitting
        
    Returns
    -------
    tuple
        (hc_results, ad_results, summary_df)
    """
    
    print("="*70)
    print("EEG 1/f SLOPE ANALYSIS")
    print("="*70)
    
    # Find all EEG files
    print("\n[1] Searching for EEG files...")
    print("-"*70)
    
    hc_files = find_eeg_files(hc_dir, 'HC')
    ad_files = find_eeg_files(ad_dir, 'AD')
    
    print(f"Found {len(hc_files)} HC subjects")
    print(f"Found {len(ad_files)} AD subjects")
    
    if len(hc_files) == 0:
        print(f"✗ No HC files found in: {hc_dir}")
        print("  Expected structure: HC/sub-XXX/eeg/*.set")
    
    if len(ad_files) == 0:
        print(f"✗ No AD files found in: {ad_dir}")
        print("  Expected structure: AD/sub-XXX/eeg/*.set")
    
    if len(hc_files) == 0 or len(ad_files) == 0:
        return None, None, None
    
    # Process HC files
    print("\n[2] Processing HC subjects...")
    print("-"*70)
    
    hc_results = []
    for i, file_info in enumerate(hc_files, 1):
        print(f"  [{i}/{len(hc_files)}] {file_info['subject_id']}: {file_info['filename']}")
        result = process_eeg_file(file_info, freq_range=freq_range)
        hc_results.append(result)
        
        if result['success']:
            print(f"      ✓ Exponent: {result['aperiodic_exponent']:.3f}, "
                  f"Offset: {result['aperiodic_offset']:.3f}, "
                  f"R²: {result['r_squared']:.3f}")
        else:
            print(f"      ✗ Failed")
    
    # Process AD files
    print("\n[3] Processing AD subjects...")
    print("-"*70)
    
    ad_results = []
    for i, file_info in enumerate(ad_files, 1):
        print(f"  [{i}/{len(ad_files)}] {file_info['subject_id']}: {file_info['filename']}")
        result = process_eeg_file(file_info, freq_range=freq_range)
        ad_results.append(result)
        
        if result['success']:
            print(f"      ✓ Exponent: {result['aperiodic_exponent']:.3f}, "
                  f"Offset: {result['aperiodic_offset']:.3f}, "
                  f"R²: {result['r_squared']:.3f}")
        else:
            print(f"      ✗ Failed")
    
    # Create summary dataframe
    all_results = hc_results + ad_results
    summary_df = pd.DataFrame(all_results)
    
    return hc_results, ad_results, summary_df


# ============================================================
# Statistical Analysis
# ============================================================

def compute_all_cohens_d(hc_results, ad_results):
    """
    Compute Cohen's d for all measures between HC and AD groups.
    
    Parameters
    ----------
    hc_results : list
        List of result dictionaries for HC subjects
    ad_results : list
        List of result dictionaries for AD subjects
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Cohen's d for each measure
    """
    
    measures = ['aperiodic_exponent', 'aperiodic_offset', 
                'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    
    cohens_results = []
    
    for measure in measures:
        # Extract values, excluding NaNs
        ad_vals = [r[measure] for r in ad_results 
                   if measure in r and not np.isnan(r[measure])]
        hc_vals = [r[measure] for r in hc_results 
                   if measure in r and not np.isnan(r[measure])]
        
        if len(ad_vals) > 1 and len(hc_vals) > 1:
            result = compute_cohens_d(ad_vals, hc_vals)
            result['measure'] = measure
            cohens_results.append(result)
    
    return pd.DataFrame(cohens_results)


# ============================================================
# Visualization Functions
# ============================================================

def plot_group_comparison(hc_results, ad_results, output_prefix='eeg_comparison'):
    """Create comprehensive visualization of group differences."""
    
    measures = ['aperiodic_exponent', 'aperiodic_offset', 
                'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    
    measure_labels = {
        'aperiodic_exponent': '1/f Exponent',
        'aperiodic_offset': '1/f Offset',
        'theta_power': 'Theta Power (4-8 Hz)',
        'alpha_power': 'Alpha Power (8-13 Hz)',
        'beta_power': 'Beta Power (13-30 Hz)',
        'gamma_power': 'Gamma Power (30-40 Hz)'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, measure in enumerate(measures):
        # Extract values
        ad_vals = [r[measure] for r in ad_results 
                   if measure in r and not np.isnan(r[measure])]
        hc_vals = [r[measure] for r in hc_results 
                   if measure in r and not np.isnan(r[measure])]
        
        if len(ad_vals) == 0 or len(hc_vals) == 0:
            axes[idx].text(0.5, 0.5, 'No valid data', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(measure_labels[measure])
            continue
        
        # Create violin plot
        parts = axes[idx].violinplot([hc_vals, ad_vals], 
                                     positions=[1, 2],
                                     showmeans=True,
                                     showextrema=True)
        
        # Color the violins
        parts['bodies'][0].set_facecolor('#A9A9A9')
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_facecolor('#90EE90')
        parts['bodies'][1].set_alpha(0.7)
        
        # Add individual points
        for i, vals in enumerate([hc_vals, ad_vals], 1):
            x = np.random.normal(i, 0.04, len(vals))
            axes[idx].scatter(x, vals, alpha=0.5, s=30, c='black', zorder=3)
        
        # Compute Cohen's d
        if len(ad_vals) > 1 and len(hc_vals) > 1:
            cohens = compute_cohens_d(ad_vals, hc_vals)
            
            # Add significance info
            axes[idx].text(0.98, 0.98, 
                          f"d = {cohens['cohens_d']:.3f}\n"
                          f"n_HC = {cohens['n_HC']}\n"
                          f"n_AD = {cohens['n_AD']}", 
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        axes[idx].set_xticks([1, 2])
        axes[idx].set_xticklabels(['HC', 'AD'])
        axes[idx].set_ylabel(measure_labels[measure], fontsize=11)
        axes[idx].set_title(measure_labels[measure], fontsize=11, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
    
    plt.suptitle('EEG Measures: HC vs AD Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{output_prefix}_violins.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_violins.pdf', bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_violins.png/pdf")
    plt.close()


def plot_cohens_d_barplot(cohens_df, output_prefix='eeg_cohens_d'):
    """Create bar plot of Cohen's d effect sizes."""
    
    measure_labels = {
        'aperiodic_exponent': '1/f Exponent',
        'aperiodic_offset': '1/f Offset',
        'theta_power': 'Theta',
        'alpha_power': 'Alpha',
        'beta_power': 'Beta',
        'gamma_power': 'Gamma'
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_pos = np.arange(len(cohens_df))
    colors = ['#E74C3C' if d < 0 else '#2ECC71' 
              for d in cohens_df['cohens_d']]
    
    bars = ax.bar(x_pos, cohens_df['cohens_d'], 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(cohens_df.iterrows()):
        height = row['cohens_d']
        ax.text(i, height, f"{height:.3f}", 
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Cohen's d (AD - HC)", fontsize=13, fontweight='bold')
    ax.set_title("Effect Sizes: AD vs HC", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([measure_labels.get(m, m) for m in cohens_df['measure']], 
                       rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add effect size interpretation lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.text(0.02, 0.2, 'small', transform=ax.get_yaxis_transform(), 
            fontsize=8, alpha=0.5, style='italic')
    ax.text(0.02, 0.5, 'medium', transform=ax.get_yaxis_transform(), 
            fontsize=8, alpha=0.5, style='italic')
    ax.text(0.02, 0.8, 'large', transform=ax.get_yaxis_transform(), 
            fontsize=8, alpha=0.5, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_barplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_barplot.pdf', bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_barplot.png/pdf")
    plt.close()


def plot_example_spectra(hc_results, ad_results, n_examples=5, output_prefix='eeg_spectra'):
    """Plot example power spectra from each group."""
    
    # Select random examples that have successful fits
    hc_valid = [r for r in hc_results if r.get('success', False) and 'fooof_model' in r]
    ad_valid = [r for r in ad_results if r.get('success', False) and 'fooof_model' in r]
    
    if len(hc_valid) == 0 or len(ad_valid) == 0:
        print("✗ No valid FOOOF models to plot")
        return
    
    n_hc = min(n_examples, len(hc_valid))
    n_ad = min(n_examples, len(ad_valid))
    
    hc_examples = np.random.choice(len(hc_valid), n_hc, replace=False)
    ad_examples = np.random.choice(len(ad_valid), n_ad, replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot HC examples
    for idx in hc_examples:
        r = hc_valid[idx]
        fm = r['fooof_model']
        axes[0].loglog(fm.freqs, fm.power_spectrum, 
                      alpha=0.5, linewidth=1.5, color='#A9A9A9')
    
    axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Power', fontsize=12)
    axes[0].set_title(f'HC Power Spectra (n={n_hc})', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot AD examples
    for idx in ad_examples:
        r = ad_valid[idx]
        fm = r['fooof_model']
        axes[1].loglog(fm.freqs, fm.power_spectrum, 
                      alpha=0.5, linewidth=1.5, color='#90EE90')
    
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Power', fontsize=12)
    axes[1].set_title(f'AD Power Spectra (n={n_ad})', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_examples.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_examples.pdf', bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_examples.png/pdf")
    plt.close()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    # ===== CONFIGURE THESE PATHS =====
    HC_DIR = "./HC"  # Path to HC folder
    AD_DIR = "./AD"  # Path to AD folder
    # ==================================
    
    # Frequency range for FOOOF fitting
    FREQ_RANGE = [1, 40]
    
    # Run the analysis
    hc_results, ad_results, summary_df = analyze_eeg_folders(
        HC_DIR, AD_DIR, freq_range=FREQ_RANGE
    )
    
    if hc_results is None or ad_results is None:
        print("\n✗ Analysis failed. Please check your directory structure.")
        exit(1)
    
    # Count successful analyses
    n_hc_success = sum(1 for r in hc_results if r.get('success', False))
    n_ad_success = sum(1 for r in ad_results if r.get('success', False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"HC: {n_hc_success}/{len(hc_results)} successful")
    print(f"AD: {n_ad_success}/{len(ad_results)} successful")
    
    # Compute Cohen's d
    print("\n[4] Computing Cohen's d...")
    print("-"*70)
    
    cohens_df = compute_all_cohens_d(hc_results, ad_results)
    
    print("\nCOHEN'S D EFFECT SIZES (AD vs HC)")
    print("="*70)
    print(cohens_df[['measure', 'cohens_d', 's_pooled', 'mean_AD', 'mean_HC']].to_string(index=False))
    
    # Save results to CSV
    print("\n[5] Saving results...")
    print("-"*70)
    
    summary_df.to_csv('eeg_analysis_summary.csv', index=False)
    print("✓ Saved eeg_analysis_summary.csv")
    
    cohens_df.to_csv('eeg_cohens_d_results.csv', index=False)
    print("✓ Saved eeg_cohens_d_results.csv")
    
    # Generate plots
    print("\n[6] Generating plots...")
    print("-"*70)
    
    plot_group_comparison(hc_results, ad_results)
    plot_cohens_d_barplot(cohens_df)
    plot_example_spectra(hc_results, ad_results, n_examples=5)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  • eeg_analysis_summary.csv - Complete results for all subjects")
    print("  • eeg_cohens_d_results.csv - Cohen's d effect sizes")
    print("  • eeg_comparison_violins.png/pdf - Violin plots comparing groups")
    print("  • eeg_cohens_d_barplot.png/pdf - Bar plot of effect sizes")
    print("  • eeg_spectra_examples.png/pdf - Example power spectra")
