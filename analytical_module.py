"""
=============================================================
ANALİTİK PERFORMANS MODÜLÜ
=============================================================
Ankara University / FS3P Lab

İçerik:
  - Kalibrasyon eğrisi (lineer regresyon)
  - LOD = 3σ/m,  LOQ = 10σ/m
  - Hassasiyet (normalize edilmiş: μA μM⁻¹ cm⁻²)
  - İntra-day ve inter-day %RSD
  - Geri kazanım (recovery) hesabı
  - Standart ilave (standard addition) metodu
  - Yayına hazır tablolar ve grafikler

Giriş CSV formatı (kalibrasyon):
  concentration_uM, ip_1, ip_2, ip_3  (tekrarlar sütunlarda)
  VEYA her tekrar için satır olarak:
  concentration_uM, ip_uA, run (tekrar numarası)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'axes.labelsize': 13,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# CSS ile senkronize ana palet — primary: #1a3a5c, secondary: #2b7cb8
COLORS = ['#1a3a5c', '#c55a11', '#375623', '#7030a0', '#843c0c',
          '#1f77b4', '#ff7f0e', '#2ca02c']

WATERMARK_TEXT  = '© SilverstoneLab'
WATERMARK_ALPHA = 0.18

def _add_watermark(fig):
    """Her sekle SilverstoneLab filigrani ekler."""
    fig.text(0.98, 0.01, WATERMARK_TEXT,
             fontsize=8, color='#1a3a5c', alpha=WATERMARK_ALPHA,
             ha='right', va='bottom', style='italic',
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════

def load_calibration_data(filepath_or_dict):
    """
    Kalibrasyon verisini yükler.

    Kabul edilen formatlar:

    FORMAT A — Geniş tablo (tekrarlar sütunlarda):
      Konsantrasyon(uM), ip_run1(uA), ip_run2(uA), ip_run3(uA)

    FORMAT B — Uzun tablo (her satır bir ölçüm):
      Konsantrasyon(uM), ip(uA), Gün

    FORMAT C — dict olarak doğrudan:
      {konsantrasyon: [ip1, ip2, ip3], ...}

    Returns
    -------
    DataFrame: 'conc_uM', 'ip_uA_mean', 'ip_uA_std', 'ip_uA_values' (list)
    """
    if isinstance(filepath_or_dict, dict):
        rows = []
        for conc, vals in filepath_or_dict.items():
            vals = np.array(vals, dtype=float)
            rows.append({
                'conc_uM':      float(conc),
                'ip_uA_mean':   vals.mean(),
                'ip_uA_std':    vals.std(ddof=1) if len(vals) > 1 else 0.0,
                'ip_uA_values': vals.tolist(),
                'n_rep':        len(vals)
            })
        return pd.DataFrame(rows).sort_values('conc_uM').reset_index(drop=True)

    df = pd.read_csv(filepath_or_dict, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]

    # FORMAT A tespiti: 3+ sütun ve sütunlardan biri "run"/"day" değilse
    conc_col = df.columns[0]
    value_cols = [c for c in df.columns[1:] if 'conc' not in c]

    rows = []
    for _, row in df.iterrows():
        vals = row[value_cols].dropna().astype(float).values
        rows.append({
            'conc_uM':      float(row[conc_col]),
            'ip_uA_mean':   vals.mean(),
            'ip_uA_std':    vals.std(ddof=1) if len(vals) > 1 else 0.0,
            'ip_uA_values': vals.tolist(),
            'n_rep':        len(vals)
        })

    return pd.DataFrame(rows).sort_values('conc_uM').reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# 2. KALİBRASYON VE LOD/LOQ
# ══════════════════════════════════════════════════════════════

def calibration_analysis(cal_data, analyte_name='Analyte', electrode_area_cm2=None,
                          unit='μM', output_dir='.', plot_style=None):
    """
    Kalibrasyon eğrisi, LOD, LOQ, hassasiyet hesabı.

    LOD = 3 × S_y_x / m
    LOQ = 10 × S_y_x / m

    Burada S_y_x = √(Σ(yi-ŷi)² / (n-2))  (regresyon standart hatası)
    Bu IUPAC tarafından önerilen en güvenilir yaklaşımdır.

    Parameters
    ----------
    cal_data        : DataFrame, load_calibration_data() çıktısı
    analyte_name    : str
    electrode_area_cm2 : float, normalize edilmiş hassasiyet için (None → atla)
    unit            : str, konsantrasyon birimi

    Returns
    -------
    dict: slope, intercept, R, R2, LOD, LOQ, sensitivity
    """
    C = cal_data['conc_uM'].values
    ip_mean = cal_data['ip_uA_mean'].values

    slope, intercept, r, p_val, se = stats.linregress(C, ip_mean)
    r_sq = r**2

    # Regresyon standart hatası (S_y_x)
    ip_pred = slope * C + intercept
    residuals = ip_mean - ip_pred
    s_yx = np.sqrt(np.sum(residuals**2) / (len(C) - 2))

    # LOD ve LOQ
    lod = 3 * s_yx / slope
    loq = 10 * s_yx / slope

    print("\n" + "═"*55)
    print(f"KALİBRASYON ANALİZİ — {analyte_name}")
    print("═"*55)
    print(f"\nLineer aralık : {C.min():.4g} – {C.max():.4g} {unit}")
    print(f"Denklem       : ip = {slope:.4f} × C + {intercept:.4f}  (μA, {unit})")
    print(f"r             : {r:.6f}")
    print(f"R²            : {r_sq:.6f}")
    print(f"Eğim (m)      : {slope:.6f} μA/{unit}")
    print(f"S_y,x         : {s_yx:.6f} μA")
    print(f"\n▶ LOD         : {lod:.4f} {unit}")
    print(f"▶ LOQ         : {loq:.4f} {unit}")

    sensitivity = slope
    if electrode_area_cm2:
        sens_normalized = slope / electrode_area_cm2
        print(f"\n▶ Hassasiyet  : {slope:.4f} μA/{unit}")
        print(f"▶ Norm. hass. : {sens_normalized:.4f} μA/{unit}·cm²")
        sensitivity = sens_normalized

    # Artıklar grafiği (residual plot)
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)
    _ls     = _ps.get('fit_linestyle', '--')
    _ms     = _ps.get('marker_size', 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Kalibrasyon eğrisi
    ax1 = axes[0]
    C_fit = np.linspace(C.min()*0.9, C.max()*1.1, 200)
    ip_fit = slope * C_fit + intercept

    # Hata çubukları
    # ip_uA_std sütunu her zaman vardır; NaN (n=1, ddof=1) → 0 ile değiştir
    _std_raw = cal_data['ip_uA_std'].values if 'ip_uA_std' in cal_data.columns else np.zeros(len(C))
    _std_vals = np.where(np.isnan(_std_raw), 0.0, _std_raw)
    _ms_pt = max(6, int(np.sqrt(_ms / np.pi) * 2))   # s→markersize dönüşümü

    if _std_vals.sum() > 0:
        ax1.errorbar(C, ip_mean, yerr=_std_vals,
                     fmt='o', color=_colors[0], capsize=5, capthick=1.5,
                     elinewidth=1.5, markersize=_ms_pt, zorder=5,
                     label='Mean ± SD')
    else:
        ax1.scatter(C, ip_mean, color=_colors[0], s=_ms, zorder=5,
                    label='Experimental (tek tekrar)')

    ax1.plot(C_fit, ip_fit, color=_colors[1 % len(_colors)], linewidth=_lw, linestyle=_ls,
             label=f'ip = {slope:.4f}C + {intercept:.4f}\nr = {r:.5f}')

    # %95 güven aralığı (prediction interval)
    # t_crit: scipy ile doğru serbestlik derecesinde hesaplanır (df = n-2)
    # Sabit 2.0 kullanmak küçük n değerlerinde hatalı (n=5: t=3.18, n=4: t=4.30)
    from scipy.stats import t as _t_dist
    n_pts   = len(C)
    df_resid = max(n_pts - 2, 1)   # sıfır bölme koruması
    t_crit  = _t_dist.ppf(0.975, df=df_resid)
    se_pred = s_yx * np.sqrt(1 + 1/n_pts + (C_fit - C.mean())**2 / np.sum((C - C.mean())**2))
    ax1.fill_between(C_fit, ip_fit - t_crit*se_pred, ip_fit + t_crit*se_pred,
                     alpha=0.12, color=_colors[1 % len(_colors)], label='95% PI')

    # LOD ve LOQ çizgileri
    ax1.axvline(lod, color='green', linestyle=':', linewidth=1.5,
                label=f'LOD = {lod:.4f} {unit}')
    ax1.axvline(loq, color='orange', linestyle=':', linewidth=1.5,
                label=f'LOQ = {loq:.4f} {unit}')

    ax1.set_xlabel(f'[{analyte_name}] ({unit})')
    ax1.set_ylabel('i$_p$ (μA)')
    ax1.set_title(f'Calibration Curve — {analyte_name}')
    ax1.legend(fontsize=8)

    # Artıklar grafiği
    ax2 = axes[1]
    ax2.scatter(C, residuals, color=_colors[2 % len(_colors)], s=_ms)
    ax2.axhline(0, color='black', linewidth=1.2)
    ax2.axhline(s_yx, color='gray', linestyle='--', linewidth=1, label=f'+S_y,x')
    ax2.axhline(-s_yx, color='gray', linestyle='--', linewidth=1, label=f'−S_y,x')
    ax2.set_xlabel(f'[{analyte_name}] ({unit})')
    ax2.set_ylabel('Residuals (μA)')
    ax2.set_title('Residual Plot\n(Random dağılım → doğrusallık OK)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'calibration_curve.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafik kaydedildi: {out_path}")

    return {
        'analyte':       analyte_name,
        'slope':         slope,
        'intercept':     intercept,
        'r':             r,
        'R2':            r_sq,
        's_yx':          s_yx,
        'LOD':           lod,
        'LOQ':           loq,
        'sensitivity':   sensitivity,
        'linear_range':  (C.min(), C.max()),
        'unit':          unit
    }

# ══════════════════════════════════════════════════════════════
# 3. %RSD (Tekrarlanabilirlik / Yeniden Üretilebilirlik)
# ══════════════════════════════════════════════════════════════

def rsd_analysis(data_dict, label='Repeatability'):
    """
    %RSD hesabı (intra-day, inter-day veya electrode-to-electrode).

    Parameters
    ----------
    data_dict : dict
        Basit örnek: {'intra_day': [15.2, 15.4, 15.1, 15.3, 15.2]}
        Çoklu gün  : {'Day 1': [15.2, 15.3], 'Day 2': [15.1, 15.4], ...}

    Returns
    -------
    DataFrame: ortalama, SD, %RSD
    """
    print(f"\n{'─'*45}")
    print(f"{label} (%RSD)")
    print(f"{'─'*45}")

    rows = []
    all_values = []

    for key, vals in data_dict.items():
        vals = np.array(vals, dtype=float)
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rsd  = (std / mean) * 100 if mean != 0 else 0.0
        all_values.extend(vals.tolist())
        rows.append({'Group': key, 'Mean (μA)': f'{mean:.4f}',
                     'SD (μA)': f'{std:.4f}', '%RSD': f'{rsd:.2f}'})
        print(f"  {key}: {mean:.4f} ± {std:.4f} μA  (%RSD = {rsd:.2f}%)")

    # Toplam %RSD
    all_arr = np.array(all_values)
    total_rsd = (all_arr.std(ddof=1) / all_arr.mean()) * 100
    print(f"  TOPLAM %RSD = {total_rsd:.2f}%")

    if total_rsd < 3:
        print("  ✓ Mükemmel (< 3%)")
    elif total_rsd < 5:
        print("  ✓ Kabul edilebilir (< 5%)")
    else:
        print("  ✗ Yüksek (> 5%) — kontrol gerekli")

    return pd.DataFrame(rows), total_rsd

# ══════════════════════════════════════════════════════════════
# 4. GERİ KAZANIM (RECOVERY)
# ══════════════════════════════════════════════════════════════

def recovery_analysis(recovery_data, slope, intercept, analyte_name='Analyte',
                       unit='μM', output_dir='.'):
    """
    Spike recovery hesabı.

    Recovery% = (C_found / C_added) × 100

    C_found, kalibrasyon eğrisinden ip → C dönüşümüyle:
    C_found = (ip_measured - intercept) / slope

    Parameters
    ----------
    recovery_data : dict veya DataFrame
        dict formatı:
        {
          'Matriksler': ['Serum', 'İdrar', 'Su'],
          'spike_uM':   [1.0, 1.0, 1.0],
          'ip_uA':      [[ip1, ip2, ip3], ...]  ← her matriksde 3 ölçüm
        }
        DataFrame: matrix, spike_uM, ip_1_uA, ip_2_uA, ip_3_uA

    Returns
    -------
    DataFrame: Matriksler, spike_uM, C_found, Recovery%, %RSD
    """
    if isinstance(recovery_data, dict):
        rows_input = []
        for mat, spike, ip_vals in zip(
            recovery_data['Matriksler'],
            recovery_data['spike_uM'],
            recovery_data['ip_uA']
        ):
            rows_input.append({
                'Matrix': mat,
                'C_added': spike,
                'ip_vals': np.array(ip_vals)
            })
    else:
        rows_input = []
        for _, row in recovery_data.iterrows():
            ip_cols = [c for c in row.index if 'ip' in c.lower()]
            rows_input.append({
                'Matrix': row.iloc[0],
                'C_added': float(row.iloc[1]),
                'ip_vals': row[ip_cols].dropna().astype(float).values
            })

    print(f"\n{'═'*65}")
    print(f"GERİ KAZANIM ANALİZİ — {analyte_name}")
    print("═"*65)
    print(f"{'Matriksler':<18} {'C_ekl.':>8} {'C_bul.':>8} {'Rec%':>8} {'%RSD':>8}")
    print(f"{'':18} {f'({unit})':>8} {f'({unit})':>8} {'':>8} {'':>8}")
    print("-"*65)

    result_rows = []
    for item in rows_input:
        ip_arr = item['ip_vals']
        c_found_arr = (ip_arr - intercept) / slope  # kalibrasyon eğrisinden geri hesap
        c_found_mean = c_found_arr.mean()
        c_found_std  = c_found_arr.std(ddof=1) if len(c_found_arr) > 1 else 0.0

        rec = (c_found_mean / item['C_added']) * 100
        rsd = (c_found_std / c_found_mean) * 100 if c_found_mean != 0 else 0.0

        print(f"{item['Matrix']:<18} {item['C_added']:>8.4f} {c_found_mean:>8.4f} "
              f"{rec:>8.2f} {rsd:>8.2f}")

        result_rows.append({
            'Matrix':          item['Matrix'],
            f'C_added ({unit})':  f"{item['C_added']:.4f}",
            f'C_found ({unit})':  f"{c_found_mean:.4f} ± {c_found_std:.4f}",
            'Recovery (%)':    f"{rec:.2f}",
            '%RSD':            f"{rsd:.2f}"
        })

    df_result = pd.DataFrame(result_rows)
    out_path = os.path.join(output_dir, 'recovery_table.csv')
    df_result.to_csv(out_path, index=False)
    print(f"\n  Tablo kaydedildi: {out_path}")
    print(f"  Hedef: Recovery 97–103%,  %RSD < 5%")

    return df_result

# ══════════════════════════════════════════════════════════════
# 5. STANDART İLAVE METODU
# ══════════════════════════════════════════════════════════════

def standard_addition(additions_uM, ip_uA, sample_volume_mL=10.0,
                       analyte_name='Analyte', output_dir='.', plot_style=None):
    """
    Standart ilave metodu ile numunedeki konsantrasyon hesabı.

    İdeal olarak bilinmeyen numune bir ölçüm alır (ip_0),
    sonra artan konsantrasyonlarda standart eklenir.

    Grafik: ip vs C_eklenen → x-eksenini kestiği nokta = −C_numune

    Parameters
    ----------
    additions_uM : array-like
        Eklenen standart konsantrasyonları (μM), 0 dahil (orijinal numune)
    ip_uA : array-like
        Her ekleme için ölçülen peak akımı (μA)
    sample_volume_mL : float
        Numune hacmi

    Returns
    -------
    dict: C_sample, grafik
    """
    C_add = np.array(additions_uM, dtype=float)
    ip    = np.array(ip_uA, dtype=float)

    slope, intercept, r, p_val, se = stats.linregress(C_add, ip)
    r_sq = r**2

    # Numune konsantrasyonu: x-eksenini kesme noktası = −C_sample
    # ip = slope × C_add + intercept = 0  →  C_add_xintercept = −intercept/slope  (negatif)
    # C_sample = |x-intercept| = intercept / slope  (pozitif)
    #
    # ⚠ DİLÜSYON DÜZELTMESİ NOTU:
    # Bu hesap, her standart eklemesinde toplam hacmin sabit kaldığını varsayar
    # (örn. çok küçük hacimde konsantre standart eklenir → hacim ihmal edilebilir).
    # Eğer eklenen standart hacimleri numune hacmine (sample_volume_mL) göre
    # ihmal edilemez düzeydeyse, C_sample gerçek konsantrasyondan yüksek çıkar.
    # Düzeltme: C_sample_gerçek = C_sample × (V_numune / (V_numune + V_ekleme_toplam))
    # Bu uygulamada hacim düzeltmesi kullanıcıya bırakılmıştır; ekleme hacimleri
    # numune hacminin <1%'i ise düzeltme ihmal edilebilir.
    C_sample = intercept / slope

    print(f"\n{'═'*55}")
    print(f"STANDART İLAVE METODU — {analyte_name}")
    print("═"*55)
    print(f"Denklem  : ip = {slope:.4f} × C + {intercept:.4f}")
    print(f"R²       : {r_sq:.6f}")
    print(f"\n▶ Numunedeki {analyte_name} = {C_sample:.4f} μM")
    print(f"   (Hacim düzeltmesi uygulanmadı — ekleme hacimleri ihmal edilebilir varsayımı)")

    # Grafik
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)
    _ls     = _ps.get('fit_linestyle', '--')
    _ms     = _ps.get('marker_size', 80)

    fig, ax = plt.subplots(figsize=(7, 5))
    C_fit = np.linspace(-C_sample*1.3, C_add.max()*1.1, 200)
    ip_fit = slope * C_fit + intercept

    ax.scatter(C_add, ip, color=_colors[0], s=_ms, zorder=5, label='Experimental')
    ax.plot(C_fit[C_fit >= 0], ip_fit[C_fit >= 0],
            color=_colors[1 % len(_colors)], linewidth=_lw)
    ax.plot(C_fit[C_fit <= 0], ip_fit[C_fit <= 0],
            color=_colors[1 % len(_colors)], linewidth=_lw,
            linestyle=_ls, alpha=0.6)

    # Kesim noktası
    ax.scatter(-C_sample, 0, color='red', s=120, zorder=6, marker='*',
               label=f'C$_{{sample}}$ = {C_sample:.4f} μM')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(-C_sample, color='red', linewidth=1, linestyle=':', alpha=0.7)

    ax.set_xlabel(f'C$_{{added}}$ ({analyte_name}) (μM)')
    ax.set_ylabel('i$_p$ (μA)')
    ax.set_title(f'Standard Addition — {analyte_name}')
    ax.legend(fontsize=9)

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'standard_addition.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Grafik: {out_path}")

    return {'C_sample_uM': C_sample, 'slope': slope, 'R2': r_sq}

# ══════════════════════════════════════════════════════════════
# 5b. ICH Q2(R2) KESİNLİK VALİDASYONU
# ══════════════════════════════════════════════════════════════

def ich_q2_precision_validation(intraday_data, interday_data=None,
                                 analyte_name='Analyte', unit='μM',
                                 output_dir='.', plot_style=None):
    """
    ICH Q2(R2) Kesinlik Validasyonu.

    Tekrar Edilebilirlik (Repeatability / Gün İçi):
        • Aynı gün, en az 3 tekrar, en az 3 konsantrasyon seviyesi (LOQ, %50, %100)
        • Kabul kriteri: %RSD ≤ %2

    Ara Kesinlik (Intermediate Precision / Günler Arası):
        • En az 3 farklı gün, her gün en az 3 tekrar
        • Kabul kriteri: %RSD ≤ %5

    Parameters
    ----------
    intraday_data : dict
        {etiket: [ip_uA_tekrar1, ip_uA_tekrar2, ...]}
        Örnek: {'Düşük (0.1 µM)': [1.52, 1.54, 1.51], 'Orta (1.0 µM)': [...]}
    interday_data : dict veya None
        {'Gün 1': [ip1, ip2, ip3], 'Gün 2': [...], ...}
        Günlük ortalamalar üzerinden toplam %RSD hesaplanır.
    analyte_name : str
    unit : str
    output_dir : str
    plot_style : dict

    Returns
    -------
    dict: intraday_results, interday_results, pass_fail
    """
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)

    INTRA_LIMIT = 2.0   # ICH Q2(R2) repeatability
    INTER_LIMIT = 5.0   # ICH Q2(R2) intermediate precision

    print(f"\n{'█'*55}")
    print(f"  ICH Q2(R2) KESİNLİK VALİDASYONU — {analyte_name}")
    print(f"{'█'*55}")

    # ── 1. Tekrar Edilebilirlik (Gün İçi) ──────────────────
    intraday_results = {}
    print(f"\n{'─'*55}")
    print("TEKRAR EDİLEBİLİRLİK (GÜN İÇİ / REPEATABILITY)")
    print(f"Kabul Kriteri: %RSD ≤ {INTRA_LIMIT}%")
    print(f"{'─'*55}")
    print(f"{'Seviye':<25} {'Ortalama (μA)':>14} {'SD (μA)':>10} {'%RSD':>8} {'Durum':>8}")
    print("-"*55)

    for label, vals in intraday_data.items():
        arr  = np.array(vals, dtype=float)
        mean = arr.mean()
        std  = arr.std(ddof=1) if len(arr) > 1 else 0.0
        rsd  = (std / mean) * 100 if mean != 0 else 0.0
        passed = rsd <= INTRA_LIMIT
        status = "✓ Geçti" if passed else "✗ Başarısız"
        print(f"{label:<25} {mean:>14.4f} {std:>10.4f} {rsd:>8.2f} {status:>8}")
        intraday_results[label] = {
            'values': arr.tolist(), 'mean': mean, 'std': std,
            'rsd': rsd, 'n': len(arr), 'pass': passed
        }

    # ── 2. Günler Arası (Ara Kesinlik) ─────────────────────
    interday_results = {}
    if interday_data:
        print(f"\n{'─'*55}")
        print("ARA KESİNLİK (GÜNLER ARASI / INTERMEDIATE PRECISION)")
        print(f"Kabul Kriteri: %RSD ≤ {INTER_LIMIT}%")
        print(f"{'─'*55}")
        print(f"{'Gün':<20} {'Ortalama (μA)':>14} {'SD (μA)':>10} {'%RSD':>8}")
        print("-"*55)

        day_means = []
        for day, vals in interday_data.items():
            arr  = np.array(vals, dtype=float)
            mean = arr.mean()
            std  = arr.std(ddof=1) if len(arr) > 1 else 0.0
            rsd  = (std / mean) * 100 if mean != 0 else 0.0
            print(f"{day:<20} {mean:>14.4f} {std:>10.4f} {rsd:>8.2f}")
            day_means.append(mean)
            interday_results[day] = {'mean': mean, 'std': std, 'rsd': rsd, 'n': len(arr)}

        # Günler arası toplam %RSD (gün ortalamalarından)
        all_means = np.array(day_means)
        total_inter_rsd = (all_means.std(ddof=1) / all_means.mean()) * 100
        inter_passed = total_inter_rsd <= INTER_LIMIT
        inter_status = "✓ Geçti" if inter_passed else "✗ Başarısız"
        print(f"{'─'*55}")
        print(f"Günler arası toplam %RSD = {total_inter_rsd:.2f}%  →  {inter_status}")
        interday_results['_total_rsd'] = total_inter_rsd
        interday_results['_pass'] = inter_passed

    # ── 3. Grafik ──────────────────────────────────────────
    n_panels = 2 if interday_data else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Sol: Gün içi RSD bar
    ax1 = axes[0]
    labels_i = list(intraday_data.keys())
    rsds_i   = [intraday_results[l]['rsd'] for l in labels_i]
    colors_i = [_colors[0] if intraday_results[l]['pass'] else '#e74c3c' for l in labels_i]
    bars = ax1.bar(range(len(labels_i)), rsds_i, color=colors_i, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.axhline(INTRA_LIMIT, color='red', linewidth=_lw, linestyle='--',
                label=f'ICH limit ({INTRA_LIMIT}%)')
    ax1.set_xticks(range(len(labels_i)))
    ax1.set_xticklabels(labels_i, rotation=25, ha='right', fontsize=10)
    ax1.set_ylabel('%RSD')
    ax1.set_title(f'Tekrar Edilebilirlik (Gün İçi)\n{analyte_name}', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, max(max(rsds_i, default=0) * 1.4, INTRA_LIMIT * 1.5))
    for bar, rsd in zip(bars, rsds_i):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{rsd:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Sağ: Günler arası RSD bar (eğer veri varsa)
    if interday_data:
        ax2 = axes[1]
        days_list = [k for k in interday_data.keys()]
        rsds_inter = [interday_results[d]['rsd'] for d in days_list]
        col_inter = '#27ae60' if interday_results['_pass'] else '#e74c3c'
        bars2 = ax2.bar(range(len(days_list)), rsds_inter,
                        color=_colors[2 % len(_colors)], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.axhline(INTER_LIMIT, color='red', linewidth=_lw, linestyle='--',
                    label=f'ICH limit ({INTER_LIMIT}%)')
        # Toplam inter-day çizgisi
        total_inter = interday_results['_total_rsd']
        ax2.axhline(total_inter, color=col_inter, linewidth=1.5, linestyle=':',
                    label=f'Toplam %RSD: {total_inter:.2f}%')
        ax2.set_xticks(range(len(days_list)))
        ax2.set_xticklabels(days_list, rotation=25, ha='right', fontsize=10)
        ax2.set_ylabel('%RSD (gün içi)')
        ax2.set_title(f'Ara Kesinlik (Günler Arası)\n{analyte_name}', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, max(max(rsds_inter, default=0) * 1.4, INTER_LIMIT * 1.5))
        for bar, rsd in zip(bars2, rsds_inter):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{rsd:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'ich_q2_precision.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Grafik: {out_path}")

    # Özet tablosu CSV
    rows_csv = []
    for label, r in intraday_results.items():
        rows_csv.append({
            'Tip': 'Gün İçi', 'Grup': label,
            'n': r['n'], 'Ortalama (μA)': round(r['mean'], 4),
            'SD (μA)': round(r['std'], 4), '%RSD': round(r['rsd'], 2),
            'ICH Limiti (%)': INTRA_LIMIT, 'Durum': 'Geçti' if r['pass'] else 'Başarısız'
        })
    if interday_data:
        for day, r in interday_results.items():
            if day.startswith('_'):
                continue
            rows_csv.append({
                'Tip': 'Günler Arası', 'Grup': day,
                'n': r['n'], 'Ortalama (μA)': round(r['mean'], 4),
                'SD (μA)': round(r['std'], 4), '%RSD': round(r['rsd'], 2),
                'ICH Limiti (%)': INTER_LIMIT,
                'Durum': 'Geçti' if r['rsd'] <= INTER_LIMIT else 'Başarısız'
            })
    df_out = pd.DataFrame(rows_csv)
    csv_path = os.path.join(output_dir, 'ich_q2_precision.csv')
    df_out.to_csv(csv_path, index=False)

    # Pass/fail özet
    intra_rsds = [v['rsd'] for v in intraday_results.values()]
    overall_intra_pass = all(v['pass'] for v in intraday_results.values())
    overall_inter_pass = interday_results.get('_pass', True) if interday_data else None

    return {
        'intraday': intraday_results,
        'interday': interday_results,
        'overall_intra_pass': overall_intra_pass,
        'overall_inter_pass': overall_inter_pass,
        'intra_max_rsd': max(intra_rsds) if intra_rsds else 0.0,
        'inter_total_rsd': interday_results.get('_total_rsd', None) if interday_data else None,
        'plot_path': out_path,
        'csv_path': csv_path
    }


# ══════════════════════════════════════════════════════════════
# 6. KOMPLE ANALİTİK PERFORMANS
# ══════════════════════════════════════════════════════════════

def _export_calibration_excel(cal_data, cal_results,
                              analyte_name='Analyte', unit='μM',
                              electrode_area_cm2=None,
                              rsd_results=None, rec_df=None,
                              output_dir='results_analytical'):
    """
    Kalibrasyon sonuçlarını detaylı çok-sayfalı Excel dosyasına aktar.

    Sayfalar
    --------
    1. Özet       – Analitik performans metrikleri tablosu
    2. Veri       – Her konsantrasyon için tüm tekrarlar + ortalama, SD, %RSD
    3. Artıklar   – Regresyon artıkları ve yüzde artıklar
    4. Grafik     – ScatterChart (hata çubukları + fit doğrusu) gömülü
    5. Geri Kaz.  – Geri kazanım tablosu (varsa)
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                      numbers)
        from openpyxl.chart import ScatterChart, Series, Reference
        from openpyxl.chart.error_bar import ErrorBars
        from openpyxl.chart.data_source import NumDataSource, NumRef
        from openpyxl.utils import get_column_letter
    except ImportError as e:
        print(f"  [Excel] openpyxl import hatası: {e}")
        return None

    xl_path = os.path.join(output_dir, 'kalibrasyon_raporu.xlsx')
    wb = Workbook()

    # ── Stil yardımcıları ──────────────────────────────────────────────────
    NAVY   = "1A3A5C"  # CSS primary ile senkronize
    ORANGE = "C55A11"
    LGRAY  = "D9E1F2"
    MGRAY  = "BDD7EE"
    WHITE  = "FFFFFF"

    def _hdr_font(bold=True, color=WHITE, size=11):
        return Font(bold=bold, color=color, name='Calibri', size=size)

    def _hdr_fill(color=NAVY):
        return PatternFill("solid", fgColor=color)

    def _sub_fill(color=LGRAY):
        return PatternFill("solid", fgColor=color)

    def _border():
        thin = Side(style='thin', color='AAAAAA')
        return Border(left=thin, right=thin, top=thin, bottom=thin)

    def _fmt_cell(ws, row, col, value, bold=False, fill=None,
                  number_format=None, align='left', color='000000'):
        c = ws.cell(row=row, column=col, value=value)
        c.font = Font(bold=bold, name='Calibri', size=10, color=color)
        if fill:
            c.fill = fill
        if number_format:
            c.number_format = number_format
        c.alignment = Alignment(horizontal=align, vertical='center', wrap_text=True)
        c.border = _border()
        return c

    # ══════════════════════════════════════════════════════════════════════
    # SAYFA 1: ÖZET
    # ══════════════════════════════════════════════════════════════════════
    ws_sum = wb.active
    ws_sum.title = "Özet"
    ws_sum.column_dimensions['A'].width = 38
    ws_sum.column_dimensions['B'].width = 28

    # Başlık
    ws_sum.merge_cells('A1:B1')
    title_cell = ws_sum['A1']
    title_cell.value = f"ANALİTİK PERFORMANS RAPORU — {analyte_name.upper()}"
    title_cell.font  = Font(bold=True, name='Calibri', size=13, color=WHITE)
    title_cell.fill  = _hdr_fill(NAVY)
    title_cell.alignment = Alignment(horizontal='center', vertical='center')
    ws_sum.row_dimensions[1].height = 24

    # Alt başlık
    ws_sum.merge_cells('A2:B2')
    ws_sum['A2'].value = f"SilverstoneLab Echem Suite  |  Konsantrasyon birimi: {unit}"
    ws_sum['A2'].font  = Font(italic=True, name='Calibri', size=9, color='555555')
    ws_sum['A2'].alignment = Alignment(horizontal='center')

    # Parametre / Değer header
    row = 3
    for col, val in enumerate(['Parametre', 'Değer'], start=1):
        _fmt_cell(ws_sum, row, col, val, bold=True,
                  fill=_hdr_fill(MGRAY), color=NAVY, align='center')
    row += 1

    slope     = cal_results['slope']
    intercept = cal_results['intercept']
    r_val     = cal_results['r']
    r2_val    = r_val ** 2
    lod       = cal_results['LOD']
    loq       = cal_results['LOQ']
    s_yx      = cal_results.get('s_yx', np.nan)
    lin_range = cal_results['linear_range']
    sens      = slope

    metrics = [
        ("Analît",                    analyte_name),
        ("Lineer aralık",             f"{lin_range[0]:.4g} – {lin_range[1]:.4g} {unit}"),
        ("Regresyon denklemi",        f"ip = {slope:.4f}·C + ({intercept:.4f})"),
        ("Eğim (slope), μA/{unit}",   slope),
        ("Kesim (intercept), μA",     intercept),
        ("Korelasyon katsayısı (r)",  r_val),
        ("R² (belirlilik katsayısı)", r2_val),
        ("Regresyon std. hatası syx", s_yx),
        (f"LOD (3σ/m), {unit}",       lod),
        (f"LOQ (10σ/m), {unit}",      loq),
        (f"Hassasiyet (μA/{unit})",   sens),
    ]
    if electrode_area_cm2:
        metrics.append((f"Norm. hassasiyet (μA/{unit}·cm²)",
                        cal_results.get('sensitivity', np.nan)))
    if rsd_results:
        for k, v in rsd_results.items():
            metrics.append((f"%RSD ({k})", f"{v:.2f}%"))

    for i, (param, val) in enumerate(metrics):
        fill = _sub_fill(LGRAY) if i % 2 == 0 else None
        _fmt_cell(ws_sum, row, 1, param, fill=fill)
        if isinstance(val, float):
            _fmt_cell(ws_sum, row, 2, val, fill=fill,
                      number_format='0.000000', align='right')
        else:
            _fmt_cell(ws_sum, row, 2, str(val), fill=fill, align='right')
        row += 1

    ws_sum.freeze_panes = 'A4'

    # ══════════════════════════════════════════════════════════════════════
    # SAYFA 2: VERİ (Genişletilmiş)
    # ══════════════════════════════════════════════════════════════════════
    ws_data = wb.create_sheet("Veri")
    ws_data.column_dimensions['A'].width = 20

    # cal_data sütunlarını bul (load_calibration_data çeşitli isimler üretebilir)
    _conc_candidates = ['concentration_uM', 'conc_uM', 'concentration', 'conc']
    conc_col = next((c for c in _conc_candidates if c in cal_data.columns), cal_data.columns[0])
    mean_col = next((c for c in ['ip_uA_mean', 'ip_mean', 'mean'] if c in cal_data.columns), None)
    std_col  = next((c for c in ['ip_uA_std',  'ip_std',  'std']  if c in cal_data.columns), None)
    n_col    = next((c for c in ['n_rep', 'n', 'count']            if c in cal_data.columns), None)

    # Tekrar sütunları (rep_1, rep_2, ... veya ip_1, ip_2, ...)
    rep_cols = [c for c in cal_data.columns
                if c.startswith('rep_') or (c.startswith('ip_') and c[-1].isdigit())]

    max_reps = len(rep_cols)
    # Başlık satırı
    headers = [f"Konsantrasyon ({unit})"]
    for r in range(1, max_reps + 1):
        headers.append(f"Tekrar {r} (μA)")
    headers += ["Ortalama (μA)", "SD (μA)", "%RSD", "n"]

    ws_data.merge_cells(f'A1:{get_column_letter(len(headers))}1')
    ws_data['A1'].value = f"KALİBRASYON VERİSİ — {analyte_name}"
    ws_data['A1'].font  = Font(bold=True, name='Calibri', size=12, color=WHITE)
    ws_data['A1'].fill  = _hdr_fill(NAVY)
    ws_data['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws_data.row_dimensions[1].height = 20

    for col_i, hdr in enumerate(headers, start=1):
        ws_data.column_dimensions[get_column_letter(col_i)].width = 16
        _fmt_cell(ws_data, 2, col_i, hdr, bold=True,
                  fill=_hdr_fill(MGRAY), color=NAVY, align='center')

    data_start_row = 3
    for i, row_s in enumerate(cal_data.itertuples(), start=0):
        r = data_start_row + i
        fill = _sub_fill(LGRAY) if i % 2 == 0 else None
        conc_val = getattr(row_s, conc_col, None)
        _fmt_cell(ws_data, r, 1, conc_val, fill=fill,
                  number_format='0.0000', align='center')

        for j, rc in enumerate(rep_cols, start=1):
            rv = getattr(row_s, rc, np.nan)
            _fmt_cell(ws_data, r, 1 + j, None if (isinstance(rv, float) and np.isnan(rv)) else rv,
                      fill=fill, number_format='0.0000', align='center')

        mean_v = getattr(row_s, mean_col, np.nan) if mean_col else np.nan
        std_v  = getattr(row_s, std_col,  np.nan) if std_col  else np.nan
        n_v    = getattr(row_s, n_col,    np.nan) if n_col    else np.nan
        rsd_v  = (std_v / mean_v * 100) if (mean_v and mean_v != 0 and not np.isnan(std_v)) else np.nan

        col_offset = 1 + max_reps
        _fmt_cell(ws_data, r, col_offset + 1, mean_v, fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_data, r, col_offset + 2, std_v,  fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_data, r, col_offset + 3, rsd_v,  fill=fill, number_format='0.00"%"', align='center')
        _fmt_cell(ws_data, r, col_offset + 4, int(n_v) if not np.isnan(n_v) else '',
                  fill=fill, align='center')

    ws_data.freeze_panes = 'A3'

    # ══════════════════════════════════════════════════════════════════════
    # SAYFA 3: ARTIKLAR
    # ══════════════════════════════════════════════════════════════════════
    ws_res = wb.create_sheet("Artıklar")
    res_headers = [f"Konsantrasyon ({unit})", "ip_ort (μA)",
                   "ip_tahmin (μA)", "Artık (μA)", "Artık (%)", "Leverage"]
    ws_res.merge_cells(f'A1:{get_column_letter(len(res_headers))}1')
    ws_res['A1'].value = f"REGRESYON ARTIKLARI — {analyte_name}"
    ws_res['A1'].font  = Font(bold=True, name='Calibri', size=12, color=WHITE)
    ws_res['A1'].fill  = _hdr_fill(NAVY)
    ws_res['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws_res.row_dimensions[1].height = 20

    for ci, hdr in enumerate(res_headers, start=1):
        ws_res.column_dimensions[get_column_letter(ci)].width = 20
        _fmt_cell(ws_res, 2, ci, hdr, bold=True,
                  fill=_hdr_fill(MGRAY), color=NAVY, align='center')

    C_arr  = cal_data[conc_col].values.astype(float)
    if mean_col:
        ip_arr = cal_data[mean_col].values.astype(float)
    else:
        ip_arr = cal_data.iloc[:, 1].values.astype(float)
    ip_fit = slope * C_arr + intercept
    n_pts  = len(C_arr)
    C_mean = C_arr.mean()
    SS_xx  = np.sum((C_arr - C_mean) ** 2)

    for i, (c_v, ip_v, fit_v) in enumerate(zip(C_arr, ip_arr, ip_fit)):
        r = 3 + i
        fill = _sub_fill(LGRAY) if i % 2 == 0 else None
        resid   = ip_v - fit_v
        resid_p = (resid / fit_v * 100) if fit_v != 0 else np.nan
        lev     = 1 / n_pts + (c_v - C_mean) ** 2 / SS_xx
        _fmt_cell(ws_res, r, 1, c_v,     fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_res, r, 2, ip_v,    fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_res, r, 3, fit_v,   fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_res, r, 4, resid,   fill=fill, number_format='0.0000', align='center')
        _fmt_cell(ws_res, r, 5, resid_p, fill=fill, number_format='0.00',   align='center')
        _fmt_cell(ws_res, r, 6, lev,     fill=fill, number_format='0.0000', align='center')

    ws_res.freeze_panes = 'A3'

    # ══════════════════════════════════════════════════════════════════════
    # SAYFA 4: GRAFİK — ScatterChart + ErrorBars
    # ══════════════════════════════════════════════════════════════════════
    ws_chart = wb.create_sheet("Grafik")
    ws_chart.column_dimensions['A'].width = 22
    ws_chart.column_dimensions['B'].width = 18
    ws_chart.column_dimensions['C'].width = 18
    ws_chart.column_dimensions['D'].width = 18
    ws_chart.column_dimensions['E'].width = 18

    # Grafik ham veri: konsantrasyon, ip_ort, SD, ip_fit, SD+, SD-
    ws_chart['A1'].value = f"Konsantrasyon ({unit})"
    ws_chart['B1'].value = "ip_ort (μA)"
    ws_chart['C1'].value = "SD (μA)"
    ws_chart['D1'].value = "ip_fit (μA)"
    ws_chart['E1'].value = "SD+ (hata üst)"
    ws_chart['F1'].value = "SD- (hata alt)"
    for ci in range(1, 7):
        c = ws_chart.cell(row=1, column=ci)
        c.font = Font(bold=True, name='Calibri', size=10, color=WHITE)
        c.fill = _hdr_fill(NAVY)
        c.alignment = Alignment(horizontal='center')

    if std_col:
        sd_arr = cal_data[std_col].values.astype(float)
    else:
        sd_arr = np.zeros(len(C_arr))
    sd_arr = np.nan_to_num(sd_arr, nan=0.0)

    for i, (c_v, ip_v, sd_v, fit_v) in enumerate(zip(C_arr, ip_arr, sd_arr, ip_fit)):
        r = i + 2
        ws_chart.cell(row=r, column=1, value=float(c_v))
        ws_chart.cell(row=r, column=2, value=float(ip_v))
        ws_chart.cell(row=r, column=3, value=float(sd_v))
        ws_chart.cell(row=r, column=4, value=float(fit_v))
        ws_chart.cell(row=r, column=5, value=float(sd_v))   # üst hata
        ws_chart.cell(row=r, column=6, value=float(sd_v))   # alt hata

    chart_rows = n_pts + 1   # son veri satırı indeksi (1-bazlı)

    # ScatterChart oluştur
    chart = ScatterChart()
    chart.title = f"{analyte_name} Kalibrasyon Eğrisi"
    chart.style = 10
    chart.x_axis.title = f"Konsantrasyon ({unit})"
    chart.y_axis.title = "Peak Akım (μA)"
    chart.x_axis.numFmt = '0.0'
    chart.y_axis.numFmt = '0.00'
    chart.width  = 20
    chart.height = 14

    # Seri 1: Deneysel noktalar (scatter, hata çubuklu)
    xvals_exp = Reference(ws_chart, min_col=1, min_row=2, max_row=chart_rows)
    yvals_exp = Reference(ws_chart, min_col=2, min_row=2, max_row=chart_rows)
    ser_exp = Series(yvals_exp, xvals_exp, title="Deneysel (ort ± SD)")
    ser_exp.marker.symbol = "circle"
    ser_exp.marker.size   = 7
    ser_exp.marker.graphicalProperties.solidFill   = "1F4E79"
    ser_exp.marker.graphicalProperties.line.solidFill = "1F4E79"
    ser_exp.graphicalProperties.line.noFill = True   # sadece işaretçi

    # Hata çubukları (SD)
    if sd_arr.sum() > 0:
        try:
            plus_ref  = NumRef(f=f"'Grafik'!$E$2:$E${chart_rows}")
            minus_ref = NumRef(f=f"'Grafik'!$F$2:$F${chart_rows}")
            plus_src  = NumDataSource(numRef=plus_ref)
            minus_src = NumDataSource(numRef=minus_ref)
            eb = ErrorBars(
                errDir='y', errValType='cust',
                plus=plus_src, minus=minus_src
            )
            ser_exp.errBars = eb
        except Exception as _eb_err:
            print(f"  [Excel] ErrorBars eklenemedi: {_eb_err}")

    chart.series.append(ser_exp)

    # Seri 2: Fit doğrusu
    xvals_fit = Reference(ws_chart, min_col=1, min_row=2, max_row=chart_rows)
    yvals_fit = Reference(ws_chart, min_col=4, min_row=2, max_row=chart_rows)
    ser_fit = Series(yvals_fit, xvals_fit, title=f"Regresyon (R²={r2_val:.4f})")
    ser_fit.marker.symbol = "none"
    ser_fit.graphicalProperties.line.solidFill = "C55A11"
    ser_fit.graphicalProperties.line.width      = 18000   # 1.5pt ×12700
    chart.series.append(ser_fit)

    # Grafik sayfasına yerleştir (H2 hücresinden başla)
    ws_chart.add_chart(chart, "H2")

    # ══════════════════════════════════════════════════════════════════════
    # SAYFA 5: GERİ KAZANIM (opsiyonel)
    # ══════════════════════════════════════════════════════════════════════
    if rec_df is not None and not rec_df.empty:
        ws_rec = wb.create_sheet("Geri Kazanım")
        ws_rec.merge_cells(f'A1:{get_column_letter(len(rec_df.columns))}1')
        ws_rec['A1'].value = f"GERİ KAZANIM ANALİZİ — {analyte_name}"
        ws_rec['A1'].font  = Font(bold=True, name='Calibri', size=12, color=WHITE)
        ws_rec['A1'].fill  = _hdr_fill(NAVY)
        ws_rec['A1'].alignment = Alignment(horizontal='center', vertical='center')
        ws_rec.row_dimensions[1].height = 20

        for ci, col_name in enumerate(rec_df.columns, start=1):
            ws_rec.column_dimensions[get_column_letter(ci)].width = 20
            _fmt_cell(ws_rec, 2, ci, col_name, bold=True,
                      fill=_hdr_fill(MGRAY), color=NAVY, align='center')

        for ri, row_s in enumerate(rec_df.itertuples(index=False), start=0):
            r = 3 + ri
            fill = _sub_fill(LGRAY) if ri % 2 == 0 else None
            for ci, v in enumerate(row_s, start=1):
                if isinstance(v, float):
                    _fmt_cell(ws_rec, r, ci, v, fill=fill, number_format='0.00', align='center')
                else:
                    _fmt_cell(ws_rec, r, ci, v, fill=fill, align='center')

        ws_rec.freeze_panes = 'A3'

    # ── Kaydet ────────────────────────────────────────────────────────────
    wb.save(xl_path)
    return xl_path


def run_analytical_performance(calibration_source, analyte_name='Analyte',
                                electrode_area_cm2=None, unit='μM',
                                rsd_data=None, recovery_data=None,
                                std_addition_data=None,
                                output_dir='results_analytical',
                                plot_style=None):
    """
    Tüm analitik performans hesaplarını çalıştır.

    Parameters
    ----------
    calibration_source : str (CSV) veya dict {conc: [ip1,ip2,...]}
    analyte_name : str
    electrode_area_cm2 : float, normalize hassasiyet için
    unit : str
    rsd_data : dict {'intra': {'Day1':[...]}, 'inter': {'Day1':[...],...}}
    recovery_data : dict
    std_addition_data : dict {'additions_uM': [...], 'ip_uA': [...]}
    output_dir : str

    Returns
    -------
    dict: tüm sonuçlar
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'█'*55}")
    print("  ANALİTİK PERFORMANS ANALİZİ BAŞLADI")
    print(f"{'█'*55}")

    # 1. Kalibrasyon
    if isinstance(calibration_source, (str,)):
        cal_data = load_calibration_data(calibration_source)
    else:
        cal_data = load_calibration_data(calibration_source)

    cal_results = calibration_analysis(
        cal_data, analyte_name=analyte_name,
        electrode_area_cm2=electrode_area_cm2, unit=unit,
        output_dir=output_dir, plot_style=plot_style
    )
    slope     = cal_results['slope']
    intercept = cal_results['intercept']

    # 2. %RSD
    rsd_results = {}
    if rsd_data:
        for rsd_type, data in rsd_data.items():
            df_rsd, total = rsd_analysis(data, label=rsd_type)
            rsd_results[rsd_type] = total
            df_rsd.to_csv(os.path.join(output_dir, f'rsd_{rsd_type}.csv'), index=False)

    # 3. Geri kazanım
    rec_df = None
    if recovery_data:
        rec_df = recovery_analysis(
            recovery_data, slope, intercept,
            analyte_name=analyte_name, unit=unit, output_dir=output_dir
        )

    # 4. Standart ilave
    std_add_results = None
    if std_addition_data:
        std_add_results = standard_addition(
            additions_uM=std_addition_data['additions_uM'],
            ip_uA=std_addition_data['ip_uA'],
            analyte_name=analyte_name,
            output_dir=output_dir, plot_style=plot_style
        )

    # Özet tablosu (yayın için)
    summary = {
        'Linear range':              f"{cal_results['linear_range'][0]:.4g}–{cal_results['linear_range'][1]:.4g} {unit}",
        'Regression equation':       f"ip = {slope:.4f}C + {intercept:.4f}",
        'Correlation coefficient (r)': f"{cal_results['r']:.6f}",
        'LOD':                       f"{cal_results['LOD']:.4f} {unit}",
        'LOQ':                       f"{cal_results['LOQ']:.4f} {unit}",
        'Sensitivity':               f"{slope:.4f} μA/{unit}",
    }
    if electrode_area_cm2:
        summary['Norm. Sensitivity'] = f"{cal_results['sensitivity']:.4f} μA/{unit}·cm²"
    for k, v in rsd_results.items():
        summary[f'%RSD ({k})'] = f"{v:.2f}%"

    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
    summary_path = os.path.join(output_dir, 'analytical_performance_summary.csv')
    summary_df.to_csv(summary_path)

    # Excel — detaylı rapor + grafik
    try:
        xl_path = _export_calibration_excel(
            cal_data, cal_results,
            analyte_name=analyte_name, unit=unit,
            electrode_area_cm2=electrode_area_cm2,
            rsd_results=rsd_results, rec_df=rec_df,
            output_dir=output_dir
        )
        if xl_path:
            print(f"  Excel raporu → {xl_path}")
    except Exception as _xl_err:
        print(f"  [Excel] hata: {_xl_err}")
        # yedek: sade yazım
        try:
            xl_path = os.path.join(output_dir, 'analytical_performance_summary.xlsx')
            with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Özet')
                cal_data.to_excel(writer, sheet_name='Kalibrasyon_Verisi', index=False)
        except Exception:
            pass

    print(f"\n{'═'*55}")
    print("ANALİTİK PERFORMANS ÖZET")
    print("═"*55)
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")
    print(f"\nTam sonuçlar → {output_dir}/")

    return {
        'calibration': cal_results,
        'rsd': rsd_results,
        'recovery': rec_df,
        'std_addition': std_add_results,
        'summary': summary
    }


# ══════════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ─── SEÇENEK A: Dict olarak veri gir ────────────────────
    calibration_data = {
        0.1:  [0.15, 0.16, 0.14],
        0.5:  [0.72, 0.74, 0.71],
        1.0:  [1.43, 1.46, 1.41],
        5.0:  [7.21, 7.18, 7.25],
        10.0: [14.4, 14.5, 14.3],
        50.0: [71.8, 72.1, 71.5],
       100.0: [143.5, 144.0, 143.0],
    }

    rsd_data = {
        'Intra-day': {
            'n=1': [14.4, 14.5, 14.3, 14.6, 14.4]  # aynı günde 5 ölçüm
        },
        'Inter-day': {
            'Day 1': [14.4, 14.5],
            'Day 2': [14.3, 14.6],
            'Day 3': [14.5, 14.4],
        }
    }

    recovery_data = {
        'Matriksler':  ['Blood serum', 'Urine', 'Tap water'],
        'spike_uM':    [10.0, 10.0, 10.0],
        'ip_uA':       [
            [14.3, 14.4, 14.5],   # serum
            [14.6, 14.7, 14.5],   # urine
            [14.4, 14.3, 14.5],   # su
        ]
    }

    std_addition_data = {
        'additions_uM': [0, 2, 4, 6, 8, 10],
        'ip_uA':        [5.2, 8.0, 10.9, 13.8, 16.7, 19.6],
    }

    results = run_analytical_performance(
        calibration_source=calibration_data,
        analyte_name='Dopamine',
        electrode_area_cm2=0.071,  # cm²
        unit='μM',
        rsd_data=rsd_data,
        recovery_data=recovery_data,
        std_addition_data=std_addition_data,
        output_dir='results_analytical'
    )
