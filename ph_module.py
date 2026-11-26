"""
=============================================================
pH TARAMASI MODÜLÜ
=============================================================
Ankara University / FS3P Lab

İçerik:
  - Pik potansiyeli (Ep) vs pH → Nernst eğimi → m/n oranı
  - Pik akımı (ip) vs pH → optimal pH belirleme
  - 59.2/n mV/pH karşılaştırması
  - Yayına hazır grafikler

Giriş CSV formatı (her pH için DPV veya CV):
  Sütunlar: Potential(V), Current(A)
  Dosya adı: ph_4.csv, ph_5.csv, ... (pH dosya adında)

VEYA tek tablo olarak:
  pH_data.csv: pH, Ep(V), ip(A)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'axes.labelsize': 13,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COLORS = ['#1f4e79', '#c55a11', '#375623', '#7030a0', '#843c0c',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

WATERMARK_TEXT  = '© SilverstoneLab'
WATERMARK_ALPHA = 0.18

def _add_watermark(fig):
    """Her sekle SilverstoneLab filigrani ekler."""
    fig.text(0.98, 0.01, WATERMARK_TEXT,
             fontsize=8, color='#1f4e79', alpha=WATERMARK_ALPHA,
             ha='right', va='bottom', style='italic',
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════════════
# 1. PİK BULMA (DPV veya CV için)
# ══════════════════════════════════════════════════════════════

def find_peak_dpv(df, peak_type='anodic'):
    """
    DPV veya CV eğrisinden pik Ep ve ip'yi bulur.

    peak_type: 'anodic' (maks), 'cathodic' (min)
    """
    E = df['Potential'].values
    I = df['Current'].values

    I_smooth = signal.savgol_filter(I, window_length=min(11, len(I)//3*2+1),
                                    polyorder=3) if len(I) > 15 else I

    if peak_type == 'anodic':
        peaks, props = signal.find_peaks(I_smooth, prominence=np.std(I_smooth)*0.3)
    else:
        peaks, props = signal.find_peaks(-I_smooth, prominence=np.std(I_smooth)*0.3)

    if len(peaks) == 0:
        # En büyük/küçük değer
        idx = np.argmax(I_smooth) if peak_type == 'anodic' else np.argmin(I_smooth)
        return E[idx], I[idx]

    best = peaks[np.argmax(props['prominences'])]
    return E[best], I[best]

# ══════════════════════════════════════════════════════════════
# 2. pH VERİSİ YÜKLEME
# ══════════════════════════════════════════════════════════════

def load_ph_data_from_files(file_dict, peak_type='anodic'):
    """
    Her pH için ayrı CSV dosyasından pH veri tablosu oluşturur.

    Parameters
    ----------
    file_dict : dict
        {pH_degeri: 'dosya.csv'}
        Örnek: {4: 'ph_4.csv', 5: 'ph_5.csv', ...}
    peak_type : str, 'anodic' veya 'cathodic'

    Returns
    -------
    DataFrame: pH, Ep_V, ip_A
    """
    rows = []
    for ph_val, path in sorted(file_dict.items()):
        try:
            df = pd.read_csv(path, comment='#')
            df.columns = [c.strip().lower() for c in df.columns]
            # Sütun adlarını normalize et
            pot_col = next((c for c in df.columns if 'pot' in c or 'volt' in c or c == 'e'), None)
            cur_col = next((c for c in df.columns if 'cur' in c or c == 'i'), None)
            if pot_col:
                df = df.rename(columns={pot_col: 'Potential', cur_col: 'Current'})
            else:
                df.columns = ['Potential', 'Current'] + list(df.columns[2:])
            df = df[['Potential', 'Current']].dropna().astype(float)

            Ep, ip = find_peak_dpv(df, peak_type=peak_type)
            rows.append({'pH': float(ph_val), 'Ep_V': Ep, 'ip_A': ip})
            print(f"  pH {ph_val}: Ep = {Ep:.4f} V, ip = {ip*1e6:.3f} μA")
        except Exception as e:
            print(f"  pH {ph_val} yüklenemedi: {e}")

    return pd.DataFrame(rows).sort_values('pH').reset_index(drop=True)

def load_ph_data_from_table(filepath):
    """
    Önceden hazırlanmış pH tablosunu yükler.

    CSV format: pH, Ep_V, ip_A   (veya ip_uA — otomatik tespit)
    """
    df = pd.read_csv(filepath, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]

    ph_col = next((c for c in df.columns if 'ph' in c), df.columns[0])
    ep_col = next((c for c in df.columns if 'ep' in c or 'pot' in c), df.columns[1])
    ip_col = next((c for c in df.columns if 'ip' in c or 'cur' in c), df.columns[2])

    df = df.rename(columns={ph_col: 'pH', ep_col: 'Ep_V', ip_col: 'ip_A'})
    df = df[['pH', 'Ep_V', 'ip_A']].dropna().astype(float)

    # μA → A dönüşümü (eğer değerler büyükse)
    if df['ip_A'].abs().max() > 0.01:  # > 10 mA ise muhtemelen μA
        df['ip_A'] = df['ip_A'] * 1e-6
        print("  NOT: ip değerleri μA → A'ya dönüştürüldü")

    return df.sort_values('pH').reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# 3. EP vs pH ANALİZİ (NERNST EĞİMİ)
# ══════════════════════════════════════════════════════════════

def analyze_ep_vs_ph(ph_data, n_electrons=2, temperature=298.15, output_dir='.',
                     plot_style=None):
    """
    Ep vs pH lineer regresyonu ve proton/elektron oranı hesabı.

    Nernst: dEp/d(pH) = −0.0592 × m/n  (25°C, V/pH birimi)

    Teorik eğimler (25°C):
      m/n = 1/1 → −59.2 mV/pH
      m/n = 1/2 → −29.6 mV/pH
      m/n = 2/2 → −59.2 mV/pH
      m/n = 2/1 → −118.4 mV/pH

    Parameters
    ----------
    ph_data    : DataFrame, pH, Ep_V, ip_A sütunları
    n_electrons: int, bilinen elektron sayısı (Nicholson veya literatürden)
    temperature: float, K
    output_dir : str

    Returns
    -------
    dict: slope, intercept, R², m_proton, n_electrons, ratio_m_n
    """
    pH  = ph_data['pH'].values
    Ep  = ph_data['Ep_V'].values * 1000  # V → mV

    slope, intercept, r, p_val, se = stats.linregress(pH, Ep)
    r_sq = r**2

    # Nernst teorik: dEp/dpH = −59.2 × m/n mV/pH (25°C)
    nernst_factor = -0.05916 * 1000  # mV/pH @ 25°C
    ratio_m_n = slope / nernst_factor  # m/n

    # Proton sayısı tahmini
    m_proton = ratio_m_n * n_electrons
    m_proton_int = round(m_proton)

    # Teorik eğim için en yakın m/n oranı
    possible_ratios = [0.5, 1.0, 1.5, 2.0]
    nearest_ratio = min(possible_ratios, key=lambda x: abs(x - ratio_m_n))
    theoretical_slope = nernst_factor * nearest_ratio

    print("\n" + "═"*55)
    print("Ep vs pH ANALİZİ (NERNST EĞİMİ)")
    print("═"*55)
    print(f"\nLineer fit: Ep = {slope:.4f} × pH + {intercept:.4f} mV")
    print(f"R²         = {r_sq:.6f}")
    print(f"Eğim       = {slope:.4f} mV/pH")
    print(f"\nNernst teorik (25°C) = −59.2 × m/n mV/pH")
    print(f"m/n hesaplanan       = {ratio_m_n:.3f}")
    print(f"Teorik en yakın m/n  = {nearest_ratio:.1f}  (eğim: {theoretical_slope:.1f} mV/pH)")
    print(f"\nElektron sayısı (n)  = {n_electrons}")
    print(f"Proton sayısı (m)    = {m_proton:.2f} ≈ {m_proton_int}")

    if abs(ratio_m_n - 1.0) < 0.15:
        print("  → Eşit sayıda proton ve elektron katılıyor (m = n)")
    elif ratio_m_n < 0.6:
        print("  → Proton katılımı elektrona oranla düşük (m < n)")
    elif ratio_m_n > 1.4:
        print("  → Proton katılımı elektrona oranla yüksek (m > n)")

    # Grafik
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)
    _ls     = _ps.get('fit_linestyle', '--')
    _ms     = _ps.get('marker_size', 80)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ep vs pH
    ax1 = axes[0]
    ph_fit = np.linspace(pH.min()-0.2, pH.max()+0.2, 100)
    ep_fit = slope * ph_fit + intercept

    ax1.scatter(pH, Ep, color=_colors[0], s=_ms, zorder=5, label='Experimental')
    ax1.plot(ph_fit, ep_fit, color=_colors[1 % len(_colors)], linewidth=_lw,
             linestyle=_ls, label=f'Fit: {slope:.2f} mV/pH\nR² = {r_sq:.5f}')

    # Teorik Nernst çizgisi
    ep_nernst = theoretical_slope * (ph_fit - pH.mean()) + Ep.mean()
    ax1.plot(ph_fit, ep_nernst, color='gray', linewidth=1.5, linestyle='--',
             label=f'Nernst teorik\n({theoretical_slope:.1f} mV/pH)', alpha=0.7)

    ax1.set_xlabel('pH')
    ax1.set_ylabel('E$_p$ (mV)')
    ax1.set_title(f'Peak Potential vs pH\nm/n = {ratio_m_n:.2f}  (m≈{m_proton_int}, n={n_electrons})')
    ax1.legend(fontsize=9)

    # ip vs pH
    ax2 = axes[1]
    ip_uA = ph_data['ip_A'].values * 1e6
    _c2 = _colors[2 % len(_colors)]
    ax2.scatter(pH, ip_uA, color=_c2, s=_ms, zorder=5)
    ax2.plot(pH, ip_uA, color=_c2, linewidth=1.5, linestyle='-', alpha=0.6)

    # Optimal pH işaretle
    opt_idx = np.argmax(np.abs(ip_uA))
    ax2.scatter(pH[opt_idx], ip_uA[opt_idx],
                color='red', s=150, zorder=6,
                marker='*', label=f'Optimal pH = {pH[opt_idx]:.1f}')
    ax2.axvline(pH[opt_idx], color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('pH')
    ax2.set_ylabel('i$_p$ (μA)')
    ax2.set_title(f'Peak Current vs pH\nOptimal pH = {pH[opt_idx]:.1f}')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'ph_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafik kaydedildi: {out_path}")

    return {
        'slope_mV_per_pH':   slope,
        'intercept_mV':      intercept,
        'R2':                r_sq,
        'ratio_m_n':         ratio_m_n,
        'm_proton':          m_proton_int,
        'n_electrons':       n_electrons,
        'optimal_pH':        pH[opt_idx],
        'theoretical_slope': theoretical_slope
    }

# ══════════════════════════════════════════════════════════════
# 4. KOMPLE pH ANALİZİ
# ══════════════════════════════════════════════════════════════

def run_ph_analysis(source, n_electrons=2, peak_type='anodic',
                    output_dir='results_ph', plot_style=None, **kwargs):
    """
    pH taraması analizini çalıştır.

    Parameters
    ----------
    source : dict veya str
        dict → {pH: 'dosya.csv'} formatında birden fazla CV/DPV dosyası
        str  → önceden hazırlanmış tablo CSV dosyası
    n_electrons : int
        Bilinen veya varsayılan elektron sayısı
    peak_type : str
        'anodic' veya 'cathodic'
    output_dir : str

    Returns
    -------
    dict: pH analiz sonuçları
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'█'*55}")
    print("  pH TARAMASI ANALİZİ BAŞLADI")
    print(f"{'█'*55}")

    if isinstance(source, dict):
        print("\nDosyalardan veri okunuyor...")
        ph_data = load_ph_data_from_files(source, peak_type=peak_type)
    else:
        print(f"\nTablodan okunuyor: {source}")
        ph_data = load_ph_data_from_table(source)

    if ph_data.empty:
        print("HATA: Veri yüklenemedi.")
        return {}

    # pH tablosunu kaydet
    ph_data_path = os.path.join(output_dir, 'ph_data.csv')
    ph_data.to_csv(ph_data_path, index=False)

    # Analiz
    results = analyze_ep_vs_ph(ph_data, n_electrons=n_electrons,
                                output_dir=output_dir, plot_style=plot_style)

    print(f"\n{'═'*55}")
    print("pH ANALİZİ ÖZET")
    print("═"*55)
    print(f"Nernst eğimi  : {results['slope_mV_per_pH']:.2f} mV/pH")
    print(f"R²            : {results['R2']:.6f}")
    print(f"m/n oranı     : {results['ratio_m_n']:.3f}")
    print(f"Proton sayısı : {results['m_proton']}")
    print(f"Optimal pH    : {results['optimal_pH']:.1f}")
    print(f"\nGrafikler → {output_dir}/")

    return results


# ══════════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # SEÇENEK 1: Her pH için ayrı CV/DPV dosyası
    file_dict = {
        4.0: 'data/ph_4.csv',
        5.0: 'data/ph_5.csv',
        6.0: 'data/ph_6.csv',
        7.0: 'data/ph_7.csv',
        7.4: 'data/ph_74.csv',
        8.0: 'data/ph_8.csv',
    }
    results = run_ph_analysis(
        source=file_dict,
        n_electrons=2,
        peak_type='anodic',
        output_dir='results_ph'
    )

    # SEÇENEK 2: Önceden hazırlanmış tablo
    # results = run_ph_analysis(
    #     source='data/ph_table.csv',   # sütunlar: pH, Ep_V, ip_A
    #     n_electrons=2,
    #     output_dir='results_ph'
    # )
