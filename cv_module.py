"""
=============================================================
CV MODÜLÜ — Siklik Voltammetri Hesapları
=============================================================
Ankara University / FS3P Lab
Elektrokimyasal Sensör Çalışmaları

İçerik:
  - Randles-Ševčík: Elektroaktif Yüzey Alanı (ECSA)
  - Hız taraması analizi (difüzyon vs adsorpsiyon)
  - log(ip) vs log(ν) slope analizi
  - ΔEp hesabı
  - Nicholson metodu: k° (elektron transfer hız sabiti)
  - Yayına hazır grafikler

Giriş CSV formatı (her tarama hızı için ayrı):
  Sütunlar: Potential(V), Current(A)
  Dosya adı: cv_10mVs.csv, cv_25mVs.csv, ... (hız dosya adında)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.signal import find_peaks, savgol_filter
import os
import warnings
warnings.filterwarnings('ignore')

# ── Stil ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = ['#1f4e79', '#c55a11', '#375623', '#7030a0', '#843c0c',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

WATERMARK_TEXT  = '© SilverstoneLab'
WATERMARK_ALPHA = 0.18   # şeffaflık: görünür ama göze batmaz


def _add_watermark(fig):
    """Her şekle SilverstoneLab filigranı ekler."""
    fig.text(0.98, 0.01, WATERMARK_TEXT,
             fontsize=8, color='#1f4e79', alpha=WATERMARK_ALPHA,
             ha='right', va='bottom', style='italic',
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════

def load_cv_files(file_dict):
    """
    CV verilerini yükler.

    Parameters
    ----------
    file_dict : dict
        {tarama_hizi_mVs: 'dosya_yolu.csv'}
        Örnek: {10: 'data/cv_10mVs.csv', 25: 'data/cv_25mVs.csv'}

    Returns
    -------
    dict: {hız: DataFrame(Potential, Current)}
    """
    data = {}
    for rate, path in file_dict.items():
        try:
            df = pd.read_csv(path, comment='#')
            # Sütun adlarını normalize et
            df.columns = [c.strip().lower() for c in df.columns]
            # Potential sütununu bul
            pot_col = next((c for c in df.columns if 'pot' in c or 'volt' in c or c == 'e'), None)
            cur_col = next((c for c in df.columns if 'cur' in c or c == 'i'), None)
            if pot_col is None or cur_col is None:
                # İlk iki sütunu kullan
                df.columns = ['Potential', 'Current'] + list(df.columns[2:])
            else:
                df = df.rename(columns={pot_col: 'Potential', cur_col: 'Current'})
            df = df[['Potential', 'Current']].dropna().astype(float)
            data[float(rate)] = df
            print(f"  ✓ {rate} mV/s yüklendi ({len(df)} nokta)")
        except Exception as e:
            print(f"  ✗ {rate} mV/s yüklenemedi: {e}")
    return data

# ══════════════════════════════════════════════════════════════
# 2. PİK BULMA
# ══════════════════════════════════════════════════════════════

def find_cv_peaks(df, smooth=True, window=11, polyorder=3):
    """
    CV eğrisinden anodik ve katodik pikleri bulur.

    Returns
    -------
    dict: {
        'Epa': anodik pik potansiyeli (V),
        'ipa': anodik pik akımı (A),
        'Epc': katodik pik potansiyeli (V),
        'ipc': katodik pik akımı (A),
        'delta_Ep': ΔEp = Epa - Epc (mV)
    }
    """
    E = df['Potential'].values
    I = df['Current'].values

    if smooth and len(I) > window:
        I_smooth = savgol_filter(I, window_length=window, polyorder=polyorder)
    else:
        I_smooth = I

    # Anodik pik (maksimum akım)
    anodic_peaks, anodic_props = find_peaks(I_smooth, prominence=np.std(I_smooth)*0.5)
    # Katodik pik (minimum akım = ters işaret maksimumu)
    cathodic_peaks, cathodic_props = find_peaks(-I_smooth, prominence=np.std(I_smooth)*0.5)

    result = {}

    if len(anodic_peaks) > 0:
        # En belirgin anodik pik
        idx = anodic_peaks[np.argmax(anodic_props['prominences'])]
        result['Epa'] = E[idx]
        result['ipa'] = I[idx]
    else:
        result['Epa'] = np.nan
        result['ipa'] = np.nan

    if len(cathodic_peaks) > 0:
        idx = cathodic_peaks[np.argmax(cathodic_props['prominences'])]
        result['Epc'] = E[idx]
        result['ipc'] = I[idx]
    else:
        result['Epc'] = np.nan
        result['ipc'] = np.nan

    if not np.isnan(result.get('Epa', np.nan)) and not np.isnan(result.get('Epc', np.nan)):
        result['delta_Ep'] = abs(result['Epa'] - result['Epc']) * 1000  # mV
        result['E_half'] = (result['Epa'] + result['Epc']) / 2          # E₁/₂ formal potential (V)
    else:
        result['delta_Ep'] = np.nan
        result['E_half'] = np.nan

    return result

# ══════════════════════════════════════════════════════════════
# 3. RANDLES-ŠEVČÍK: ECSA HESABI
# ══════════════════════════════════════════════════════════════

def randles_sevcik_ecsa(cv_data, concentration_mM=1.0, diffusion_coeff=7.6e-6,
                         n_electrons=1, use_anodic=True, output_dir='.'):
    """
    Randles-Ševčík denklemiyle Elektroaktif Yüzey Alanı (ECSA) hesabı.

    ip = 2.69×10⁵ × n^(3/2) × A × D^(1/2) × C × ν^(1/2)

    Parameters
    ----------
    cv_data : dict
        {hız_mVs: DataFrame} — load_cv_files() çıktısı
    concentration_mM : float
        Redoks probunun konsantrasyonu (mM). Varsayılan: 1.0 mM [Fe(CN)₆]³⁻
    diffusion_coeff : float
        Difüzyon katsayısı (cm²/s). Varsayılan: 7.6×10⁻⁶ cm²/s ([Fe(CN)₆]³⁻/⁴⁻)
    n_electrons : int
        Elektron sayısı
    use_anodic : bool
        True → anodik pik, False → katodik pik
    output_dir : str
        Grafik kayıt dizini

    Returns
    -------
    dict: ECSA ve lineer fit parametreleri
    """
    C = concentration_mM * 1e-3 * 1e-3  # mM → mol/cm³

    scan_rates_Vs = []
    peak_currents_A = []

    print("\n" + "═"*50)
    print("RANDLES-ŠEVČÍK ANALİZİ")
    print("═"*50)
    print(f"{'Hız (mV/s)':<12} {'ipa (μA)':<12} {'ipc (μA)':<12} {'ΔEp (mV)':<12} {'E½ (V)':<10}")
    print("-"*60)

    peaks_table = []
    for rate in sorted(cv_data.keys()):
        df = cv_data[rate]
        p = find_cv_peaks(df)
        ipa_uA = p['ipa'] * 1e6 if not np.isnan(p.get('ipa', np.nan)) else np.nan
        ipc_uA = p['ipc'] * 1e6 if not np.isnan(p.get('ipc', np.nan)) else np.nan
        dEp = p.get('delta_Ep', np.nan)
        e_half_print = p.get('E_half', np.nan)
        e_half_str = f"{e_half_print:.4f}" if not np.isnan(e_half_print) else "—"
        print(f"{rate:<12.1f} {ipa_uA:<12.3f} {ipc_uA:<12.3f} {dEp:<12.2f} {e_half_str:<10}")

        ip = p['ipa'] if use_anodic else abs(p['ipc'])
        if not np.isnan(ip):
            scan_rates_Vs.append(rate / 1000)  # mV/s → V/s
            peak_currents_A.append(ip)
        e_half = p.get('E_half', np.nan)
        peaks_table.append({
            'scan_rate_mVs': rate,
            'Epa_V': p.get('Epa', np.nan),
            'ipa_uA': ipa_uA,
            'Epc_V': p.get('Epc', np.nan),
            'ipc_uA': ipc_uA,
            'delta_Ep_mV': dEp,
            'E_half_V': e_half
        })

    nu = np.array(scan_rates_Vs)
    ip = np.array(peak_currents_A)
    nu_sqrt = np.sqrt(nu)

    # Lineer fit: ip = m × ν^(1/2)
    slope, intercept, r, p_val, se = stats.linregress(nu_sqrt, ip)
    r_sq = r**2

    # ECSA = slope / (2.69e5 × n^(3/2) × D^(1/2) × C)
    ecsa_cm2 = slope / (2.69e5 * (n_electrons**1.5) * np.sqrt(diffusion_coeff) * C)

    print(f"\n{'─'*50}")
    print(f"Lineer fit: ip = {slope*1e6:.4f}×ν^(1/2) + {intercept*1e6:.4f}  μA")
    print(f"R² = {r_sq:.6f}")
    print(f"Eğim (slope) = {slope:.6e} A·(V/s)^(-1/2)")
    print(f"\n▶ ECSA = {ecsa_cm2:.4f} cm²")
    print(f"  (n={n_electrons}, D={diffusion_coeff:.2e} cm²/s, C={concentration_mM} mM)")

    # ── Grafik 1: ip vs ν^(1/2) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sol: CV eğrileri (tüm hızlar)
    ax1 = axes[0]
    colors_cv = plt.cm.viridis(np.linspace(0.1, 0.9, len(cv_data)))
    for i, rate in enumerate(sorted(cv_data.keys())):
        df = cv_data[rate]
        ax1.plot(df['Potential'], df['Current']*1e6,
                 color=colors_cv[i], label=f'{rate} mV/s', linewidth=1.5)
    ax1.set_xlabel('Potential (V)')
    ax1.set_ylabel('Current (μA)')
    ax1.set_title('Cyclic Voltammograms')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Sağ: ip vs ν^(1/2)
    ax2 = axes[1]
    nu_sqrt_fit = np.linspace(nu_sqrt.min()*0.9, nu_sqrt.max()*1.1, 100)
    ip_fit = slope * nu_sqrt_fit + intercept

    ax2.scatter(nu_sqrt * 1000, ip * 1e6,
                color=COLORS[0], s=80, zorder=5, label='Experimental')
    ax2.plot(nu_sqrt_fit * 1000, ip_fit * 1e6,
             color=COLORS[1], linewidth=2,
             label=f'Linear fit\ny = {slope*1e6:.4f}x + {intercept*1e6:.4f}\nR² = {r_sq:.5f}')
    ax2.set_xlabel('ν$^{1/2}$ (mV/s)$^{1/2}$')
    ax2.set_ylabel('i$_p$ (μA)')
    ax2.set_title(f'Randles-Ševčík Plot\nECSA = {ecsa_cm2:.4f} cm²')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'cv_randles_sevcik.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafik kaydedildi: {out_path}")

    return {
        'ECSA_cm2': ecsa_cm2,
        'slope': slope,
        'intercept': intercept,
        'R2': r_sq,
        'peaks_table': pd.DataFrame(peaks_table)
    }

# ══════════════════════════════════════════════════════════════
# 4. TARAMA HIZI ANALİZİ: DİFÜZYON vs ADSORPSIYON
# ══════════════════════════════════════════════════════════════

def scan_rate_mechanism(cv_data, use_anodic=True, output_dir='.'):
    """
    log(ip) vs log(ν) analizi ile kontrol mekanizmasını belirler.

    İdeal eğimler:
      b = 0.5  → Difüzyon kontrollü
      b = 1.0  → Adsorpsiyon (yüzey) kontrollü
      0.5 < b < 1.0 → Karışık

    Returns
    -------
    dict: slope (b), R², mekanizma yorumu
    """
    rates = []
    ip_vals = []

    for rate in sorted(cv_data.keys()):
        df = cv_data[rate]
        p = find_cv_peaks(df)
        ip = p['ipa'] if use_anodic else abs(p['ipc'])
        if not np.isnan(ip) and ip > 0:
            rates.append(rate / 1000)  # V/s
            ip_vals.append(ip)

    log_nu = np.log10(np.array(rates))
    log_ip = np.log10(np.array(ip_vals))

    slope, intercept, r, p_val, se = stats.linregress(log_nu, log_ip)
    r_sq = r**2

    if slope < 0.6:
        mechanism = "Difüzyon Kontrollü (b ≈ 0.5)"
        color = COLORS[0]
    elif slope > 0.85:
        mechanism = "Adsorpsiyon Kontrollü (b ≈ 1.0)"
        color = COLORS[1]
    else:
        mechanism = f"Karışık Kontrol (b = {slope:.2f})"
        color = COLORS[2]

    print("\n" + "═"*50)
    print("TARAMA HIZI MEKANİZMA ANALİZİ")
    print("═"*50)
    print(f"log(ip) = {slope:.4f} × log(ν) + {intercept:.4f}")
    print(f"b (eğim) = {slope:.4f}")
    print(f"R²       = {r_sq:.6f}")
    print(f"Sonuç    : {mechanism}")

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    log_nu_fit = np.linspace(log_nu.min()-0.1, log_nu.max()+0.1, 100)
    log_ip_fit = slope * log_nu_fit + intercept

    ax1 = axes[0]
    ax1.scatter(log_nu, log_ip, color=color, s=80, zorder=5, label='Experimental')
    ax1.plot(log_nu_fit, log_ip_fit, color='black', linewidth=2,
             label=f'b = {slope:.4f}\nR² = {r_sq:.5f}')
    ax1.set_xlabel('log(ν)  [ν: V/s]')
    ax1.set_ylabel('log(i$_p$)  [i$_p$: A]')
    ax1.set_title(f'Mekanizma Analizi\n{mechanism}')
    ax1.legend()

    # Referans çizgileri
    for b_ref, label, ls in [(0.5, 'b=0.5 (difüzyon)', '--'),
                               (1.0, 'b=1.0 (adsorpsiyon)', ':')]:
        yref = b_ref * log_nu_fit + (log_ip.mean() - b_ref * log_nu.mean())
        ax1.plot(log_nu_fit, yref, color='gray', linestyle=ls,
                 linewidth=1, alpha=0.6, label=label)
    ax1.legend(fontsize=9)

    # ip vs ν (adsorpsiyon kontrolü için doğrusal olmalı)
    ax2 = axes[1]
    nu_arr = np.array(rates) * 1000  # mV/s
    ip_arr = np.array(ip_vals) * 1e6  # μA
    slope2, int2, r2, _, _ = stats.linregress(nu_arr, ip_arr)
    ax2.scatter(nu_arr, ip_arr, color=COLORS[0], s=80, zorder=5)
    nu_fit = np.linspace(nu_arr.min()*0.9, nu_arr.max()*1.1, 100)
    ax2.plot(nu_fit, slope2*nu_fit + int2, color=COLORS[1], linewidth=2,
             label=f'R² = {r2**2:.4f}')
    ax2.set_xlabel('ν (mV/s)')
    ax2.set_ylabel('i$_p$ (μA)')
    ax2.set_title('i$_p$ vs ν  (adsorpsiyon kontrolü için doğrusal)')
    ax2.legend()

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'cv_mechanism_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik kaydedildi: {out_path}")

    return {
        'slope_b': slope,
        'R2': r_sq,
        'mechanism': mechanism
    }

# ══════════════════════════════════════════════════════════════
# 5. NİCHOLSON METODU: k° (Elektron Transfer Hız Sabiti)
# ══════════════════════════════════════════════════════════════

def nicholson_k0(cv_data, diffusion_coeff=7.6e-6, n_electrons=1,
                  temperature=298.15, output_dir='.'):
    """
    Nicholson (1965) metodu ile k° hesabı.

    ΔEp → ψ (Nicholson parametresi) → k°

    k° = ψ × (π × D × n × F × ν / R × T)^(1/2)

    Not: Yalnızca tersinir/yarı-tersinir sistemler için geçerlidir.
    Tersinir limit: ΔEp ≈ 59/n mV (25°C)

    Returns
    -------
    dict: {hız: k°} ve ortalama k°
    """
    F = 96485    # C/mol
    R = 8.314    # J/(mol·K)
    T = temperature

    # Nicholson tablosundan ψ-ΔEp ilişkisi (25°C, n=1)
    psi_table = np.array([
        20.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5,
        3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.50,
        0.35, 0.25, 0.10
    ])
    dEp_table = np.array([
        61, 65, 68, 72, 75, 80, 84,
        92, 105, 121, 141, 155, 174, 197, 228,
        256, 280, 338
    ]) / 1000  # V

    print("\n" + "═"*50)
    print("NİCHOLSON METODU — k° HESABI")
    print("═"*50)
    print(f"{'Hız (mV/s)':<12} {'ΔEp (mV)':<12} {'ψ':<10} {'k° (cm/s)':<15}")
    print("-"*50)

    k0_values = []
    results = {}

    for rate in sorted(cv_data.keys()):
        df = cv_data[rate]
        p = find_cv_peaks(df)
        dEp_V = p.get('delta_Ep', np.nan) / 1000  # mV → V

        if np.isnan(dEp_V) or dEp_V < 0.059/n_electrons:
            # Tersinir limite yakın, k° çok yüksek
            print(f"{rate:<12.1f} {'<59':<12} {'—':<10} {'>> 1 (reversible)':<15}")
            continue

        # ψ interpolasyonu
        if dEp_V > dEp_table.max():
            print(f"{rate:<12.1f} {dEp_V*1000:<12.1f} {'—':<10} {'Quasi-irreversible':<15}")
            continue

        psi = np.interp(dEp_V, dEp_table, psi_table)

        # k° = ψ × √(π × D × n × F × ν / R × T)
        nu_Vs = rate / 1000
        k0 = psi * np.sqrt(np.pi * diffusion_coeff * n_electrons * F * nu_Vs / (R * T))

        print(f"{rate:<12.1f} {dEp_V*1000:<12.1f} {psi:<10.4f} {k0:<15.6f}")
        k0_values.append(k0)
        results[rate] = k0

    if k0_values:
        k0_mean = np.mean(k0_values)
        k0_std = np.std(k0_values)
        print(f"{'─'*50}")
        print(f"Ortalama k° = {k0_mean:.6f} ± {k0_std:.6f} cm/s")
        results['mean'] = k0_mean
        results['std'] = k0_std
    else:
        print("Nicholson için uygun ΔEp aralığı bulunamadı.")

    return results

# ══════════════════════════════════════════════════════════════
# 6. KOMPLE CV ANALİZİ (hepsini çalıştır)
# ══════════════════════════════════════════════════════════════

def run_cv_analysis(file_dict, concentration_mM=1.0, diffusion_coeff=7.6e-6,
                    n_electrons=1, output_dir='results_cv'):
    """
    Tüm CV analizlerini çalıştıran ana fonksiyon.

    Parameters
    ----------
    file_dict : dict
        {tarama_hizi_mVs: 'dosya.csv'}
    concentration_mM : float
        Redoks probu konsantrasyonu (mM)
    diffusion_coeff : float
        Difüzyon katsayısı (cm²/s)
    n_electrons : int
        Elektron sayısı
    output_dir : str
        Grafiklerin kaydedileceği dizin

    Returns
    -------
    dict: Tüm sonuçlar
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'█'*55}")
    print("  CV ANALİZİ BAŞLADI")
    print(f"{'█'*55}")

    # Veri yükle
    cv_data = load_cv_files(file_dict)
    if not cv_data:
        print("HATA: Hiç veri yüklenemedi.")
        return {}

    # ECSA
    ecsa_results = randles_sevcik_ecsa(
        cv_data, concentration_mM, diffusion_coeff, n_electrons,
        output_dir=output_dir
    )

    # Mekanizma
    mech_results = scan_rate_mechanism(cv_data, output_dir=output_dir)

    # k° (Nicholson)
    k0_results = nicholson_k0(cv_data, diffusion_coeff, n_electrons,
                               output_dir=output_dir)

    # Özet tablo
    peaks_df = ecsa_results['peaks_table']
    peaks_path = os.path.join(output_dir, 'cv_peaks_summary.csv')
    peaks_df.to_csv(peaks_path, index=False)

    print(f"\n{'═'*55}")
    print("CV ANALİZİ ÖZET")
    print(f"{'═'*55}")
    print(f"ECSA              : {ecsa_results['ECSA_cm2']:.4f} cm²")
    print(f"Randles-Ševčík R² : {ecsa_results['R2']:.6f}")
    print(f"Kontrol mekaniği  : {mech_results['mechanism']}")
    print(f"b (log-log eğim)  : {mech_results['slope_b']:.4f}")
    if 'mean' in k0_results:
        print(f"k° (Nicholson)    : {k0_results['mean']:.6f} ± {k0_results['std']:.6f} cm/s")
    print(f"\nGrafikler ve tablo → {output_dir}/")

    return {
        'cv_data': cv_data,
        'ecsa': ecsa_results,
        'mechanism': mech_results,
        'k0': k0_results
    }


# ══════════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ─── Kendi dosyalarınla değiştir ───────────────────────
    file_dict = {
        10:  'data/cv_10mVs.csv',
        25:  'data/cv_25mVs.csv',
        50:  'data/cv_50mVs.csv',
        75:  'data/cv_75mVs.csv',
        100: 'data/cv_100mVs.csv',
        150: 'data/cv_150mVs.csv',
        200: 'data/cv_200mVs.csv',
    }

    results = run_cv_analysis(
        file_dict=file_dict,
        concentration_mM=1.0,          # 1 mM [Fe(CN)6]
        diffusion_coeff=7.6e-6,         # cm²/s
        n_electrons=1,
        output_dir='results_cv'
    )
