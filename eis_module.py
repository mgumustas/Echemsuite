"""
=============================================================
EIS MODÜLÜ — Elektrokimyasal Empedans Spektroskopisi
=============================================================
Ankara University / FS3P Lab

İçerik:
  - Nyquist ve Bode grafikleri
  - Randles devresi fitting: Rs + Rct//CPE//Warburg
  - Parametre tablosu (her modifikasyon adımı için)
  - Rct karşılaştırma grafiği
  - Yüzey kapasitansı ve pürüzlülük faktörü

Giriş CSV formatı:
  Sütunlar: Freq(Hz), Z_real(Ohm), Z_imag(Ohm)
  NOT: Z_imag genellikle negatif (-Im) olarak verilir.
       Pozitif olarak girilirse sign_zimag=-1 kullan.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
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
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

WATERMARK_TEXT  = '© SilverstoneLab'
WATERMARK_ALPHA = 0.18


def _add_watermark(fig):
    """Her şekle SilverstoneLab filigranı ekler."""
    fig.text(0.98, 0.01, WATERMARK_TEXT,
             fontsize=8, color='#1a3a5c', alpha=WATERMARK_ALPHA,
             ha='right', va='bottom', style='italic',
             transform=fig.transFigure)


# ══════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════

def load_eis_file(filepath, sign_zimag=-1):
    """
    EIS verisini yükler.

    Parameters
    ----------
    filepath : str
        CSV dosya yolu (Freq, Z_real, Z_imag sütunları)
    sign_zimag : int
        +1: Z_imag zaten pozitif (nadiren)
        -1: Z_imag negatif olarak verilmiş (Nyquist için pozitife çevir)
              Bu, potansiyostatların büyük çoğunluğu için doğrudur.

    Returns
    -------
    DataFrame: freq, Zr (Ohm), Zi_pos (Ohm, pozitif olarak)
    """
    df = pd.read_csv(filepath, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]

    freq_col = next((c for c in df.columns if 'freq' in c or c == 'f'), None)
    zr_col   = next((c for c in df.columns if 'real' in c or "z'" in c or "zr" in c), None)
    zi_col   = next((c for c in df.columns if 'imag' in c or 'z"' in c or 'zi' in c), None)

    if freq_col is None:
        # İlk üç sütunu kullan
        df.columns = ['freq', 'Zr', 'Zi'] + list(df.columns[3:])
    else:
        df = df.rename(columns={freq_col: 'freq', zr_col: 'Zr', zi_col: 'Zi'})

    df = df[['freq', 'Zr', 'Zi']].dropna().astype(float)
    df = df.sort_values('freq', ascending=False).reset_index(drop=True)

    # Nyquist için −Im(Z) pozitif olmalı
    df['Zi_pos'] = sign_zimag * df['Zi']  # işareti düzelt

    return df

# ══════════════════════════════════════════════════════════════
# 2. EŞDEĞİŞ DEVRE: RANDLES (Rs + Rct//CPE//Warburg)
# ══════════════════════════════════════════════════════════════

def randles_cpe_impedance(omega, Rs, Rct, Q, n_cpe, sigma):
    """
    Randles devresi empedansı (CPE ile):

    Z(ω) = Rs + [ Y_CPE + 1/(Rct + Z_W) ]^(-1)

    Y_CPE  = Q × (jω)^n
    Z_W    = σ/√ω × (1−j)   [sonsuz Warburg, ω→düşük frekanslarda geçerli]

    Parameters
    ----------
    omega : array, rad/s cinsinden açısal frekans
    Rs    : float, çözelti direnci (Ω)
    Rct   : float, yük transferi direnci (Ω)
    Q     : float, CPE katsayısı (S·s^n / cm²)
    n_cpe : float, CPE üssü (0–1; 1→kondansatör, 0.5→Warburg)
    sigma : float, Warburg katsayısı (Ω·s^(-1/2))
    """
    j = 1j
    Z_W    = sigma / np.sqrt(omega) * (1 - j)
    Y_CPE  = Q * (j * omega)**n_cpe
    Z_p    = 1 / (Y_CPE + 1 / (Rct + Z_W))
    return Rs + Z_p

def _residuals_flat(params, omega, Zr_exp, Zi_exp):
    """Fit için birleştirilmiş artıklar (gerçek + imajiner)."""
    Rs, Rct, Q, n_cpe, sigma = params
    Z_calc = randles_cpe_impedance(omega, Rs, Rct, Q, n_cpe, sigma)
    res_r = Zr_exp - Z_calc.real
    res_i = Zi_exp - Z_calc.imag
    return np.concatenate([res_r, res_i])

def fit_eis(df, label='Electrode', output_dir='.', p0=None):
    """
    Randles-CPE devresini EIS verisine uydurur.

    Parameters
    ----------
    df      : DataFrame, load_eis_file() çıktısı
    label   : str, grafik başlığı için
    p0      : list [Rs0, Rct0, Q0, n0, sigma0] — başlangıç tahminleri
              None ise otomatik tahmin edilir.

    Returns
    -------
    dict: {Rs, Rct, Q, n_cpe, sigma, chi2, label}
    """
    omega = 2 * np.pi * df['freq'].values
    Zr    = df['Zr'].values
    Zi    = -df['Zi_pos'].values   # fitting için orijinal işaret

    # Otomatik başlangıç tahmini
    if p0 is None:
        Rs0    = Zr[np.argmax(df['freq'])]   # yüksek frekansta Zr ≈ Rs
        Rct0   = (Zr.max() - Zr.min()) * 1.2
        Q0     = 1e-6
        n0     = 0.9
        sigma0 = 50.0
        p0 = [Rs0, Rct0, Q0, n0, sigma0]

    bounds = (
        # n_CPE alt sınırı 0.5→0.3: saf difüzyon (n=0.5) davranışı ve
        # yüksek yüzey heterojenliği gösteren sistemler kapsanır (Hirschorn 2010).
        [0,    0,    1e-10, 0.3,  0],
        [1e6,  1e8,  1e-2,  1.0,  1e6]
    )

    try:
        result = optimize.least_squares(
            _residuals_flat, p0,
            args=(omega, Zr, Zi),
            bounds=bounds, method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=50000
        )
        Rs_f, Rct_f, Q_f, n_f, sigma_f = result.x

        # ── Parametre belirsizlikleri (kovaryans matrisi) ──────────────────────
        # least_squares Jacobian'ından standart hatalar:  σ_i = √(diag(C))
        #   C = (J^T J)^{-1} × MSE,  MSE = Σres² / (N_obs - N_params)
        # Rank-deficient durumlarda (sinyalsiz parametreler) NaN döndürülür.
        J = result.jac
        n_obs    = len(result.fun)
        n_params = len(result.x)
        mse      = np.sum(result.fun**2) / max(n_obs - n_params, 1)
        try:
            cov  = np.linalg.inv(J.T @ J) * mse
            perr = np.sqrt(np.abs(np.diag(cov)))   # abs: sayısal gürültü ≥ 0
        except np.linalg.LinAlgError:
            perr = np.full(n_params, np.nan)

        # ── χ² (modulus weighting — EIS literatür standardı) ──────────────────
        # Her frekans noktası kendi |Z_exp|² ile normalleştirilir.
        # Eski: ortalama |Z| ile normalizasyon → düşük-frekans veriye yanlı.
        # Yeni: Σ[(ΔZr²+ΔZi²)/|Z_exp|²] / (2N - p)   [Orazem & Tribollet 2008]
        Z_fit  = randles_cpe_impedance(omega, Rs_f, Rct_f, Q_f, n_f, sigma_f)
        Zmag   = np.sqrt(Zr**2 + Zi**2)
        chi2   = (np.sum((Zr - Z_fit.real)**2 / Zmag**2 +
                         (Zi - Z_fit.imag)**2 / Zmag**2) /
                  max(2*len(omega) - n_params, 1))

        params = {
            'label':  label,
            'Rs':     Rs_f,
            'Rct':    Rct_f,
            'Q':      Q_f,
            'n_cpe':  n_f,
            'sigma':  sigma_f,
            'chi2':   chi2,
            # Parametre standart hataları [Rs_err, Rct_err, Q_err, n_err, sigma_err]
            'perr':   perr,
            'Z_fit':  Z_fit,
            'omega':  omega,
            'df':     df
        }

        # Yüzey kapasitansı (CPE → C_eff dönüşümü)
        # Randles devresi için doğru formül (Brug et al., JECS 1984):
        #   C_eff = Q^(1/n) × (1/Rs + 1/Rct)^((n-1)/n)
        # NOT: Brug'un basit formülü [Q^(1/n) × Rct^((1-n)/n)] sadece
        # Rs//CPE topolojisi için geçerlidir; Randles devresinde hatalı sonuç verir.
        C_eff = Q_f**(1/n_f) * (1/Rs_f + 1/Rct_f)**((n_f - 1)/n_f)
        params['C_eff_F'] = C_eff

        return params

    except Exception as e:
        print(f"  Fit hatası ({label}): {e}")
        return None

# ══════════════════════════════════════════════════════════════
# 3. NYQUIST VE BODE GRAFİKLERİ
# ══════════════════════════════════════════════════════════════

def plot_nyquist_comparison(fit_results, output_dir='.', filename='eis_nyquist.png',
                            plot_style=None):
    """
    Birden fazla elektrot/adım için Nyquist karşılaştırması.

    Parameters
    ----------
    fit_results : list of dict (fit_eis() çıktıları)
    plot_style  : dict, optional — colors, linewidth, fit_linestyle, marker_size
    """
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)
    _ls     = _ps.get('fit_linestyle', '--')
    _ms     = _ps.get('marker_size', 40)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, res in enumerate(fit_results):
        if res is None:
            continue
        df  = res['df']
        color = _colors[i % len(_colors)]

        # Deneysel
        ax.scatter(df['Zr'], df['Zi_pos'],
                   color=color, s=_ms, label=f"{res['label']} (exp)", zorder=4)
        # Fit
        Z_fit = res['Z_fit']
        ax.plot(Z_fit.real, -Z_fit.imag,
                color=color, linewidth=_lw, linestyle=_ls,
                label=f"{res['label']} (fit)")

    ax.set_xlabel("Z$_{real}$ (Ω)")
    ax.set_ylabel("−Z$_{imag}$ (Ω)")
    ax.set_title("Nyquist Plots — EIS")
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Nyquist grafiği: {out_path}")

def plot_bode(fit_results, output_dir='.', filename='eis_bode.png', plot_style=None):
    """
    Bode grafikleri: |Z| vs frekans ve faz açısı vs frekans.
    Her elektrota/adıma ait deneysel ve fit eğrileri.
    """
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)
    _lw     = _ps.get('linewidth', 2.0)
    _ls     = _ps.get('fit_linestyle', '--')
    _ms_bode = max(10, int(_ps.get('marker_size', 40) * 0.5))  # Bode uses smaller markers

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_mag, ax_phase = axes

    for i, res in enumerate(fit_results):
        if res is None:
            continue
        df    = res['df']
        color = _colors[i % len(_colors)]
        freq  = df['freq'].values
        Zr    = df['Zr'].values
        Zi    = -df['Zi_pos'].values   # original sign for phase

        # Magnitude |Z|
        Z_mag_exp   = np.sqrt(Zr**2 + Zi**2)
        # Phase (degrees)
        phase_exp   = np.degrees(np.arctan2(Zi, Zr))

        ax_mag.scatter(freq, Z_mag_exp, color=color, s=_ms_bode,
                       label=f"{res['label']} (exp)", zorder=4)
        ax_phase.scatter(freq, phase_exp, color=color, s=_ms_bode,
                         label=f"{res['label']} (exp)", zorder=4)

        # Fit eğrisi
        omega = res['omega']
        Z_fit = res['Z_fit']
        freq_fit = omega / (2 * np.pi)
        Z_mag_fit  = np.abs(Z_fit)
        phase_fit  = np.degrees(np.angle(Z_fit))

        ax_mag.plot(freq_fit, Z_mag_fit, color=color, linewidth=_lw,
                    linestyle=_ls, label=f"{res['label']} (fit)")
        ax_phase.plot(freq_fit, phase_fit, color=color, linewidth=_lw,
                      linestyle=_ls)

    for ax in [ax_mag, ax_phase]:
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')

    ax_mag.set_yscale('log')
    ax_mag.set_ylabel('|Z| (Ω)')
    ax_mag.set_title('Bode Plot — Impedance Magnitude')
    ax_mag.legend(fontsize=8)

    ax_phase.set_ylabel('Phase Angle (°)')
    ax_phase.set_title('Bode Plot — Phase Angle')
    ax_phase.legend(fontsize=8)

    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bode grafiği: {out_path}")


def plot_rct_comparison(fit_results, output_dir='.', plot_style=None):
    """Rct değerlerini modifikasyon adımlarına göre çubuk grafik."""
    _ps = plot_style or {}
    _colors = _ps.get('colors', COLORS)

    labels = [r['label'] for r in fit_results if r is not None]
    rcts   = [r['Rct']   for r in fit_results if r is not None]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.5), 5))
    bars = ax.bar(labels, rcts, color=_colors[:len(labels)],
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, rcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rcts)*0.01,
                f'{val:.1f} Ω', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("R$_{ct}$ (Ω)")
    ax.set_title("Charge Transfer Resistance — Modification Steps")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    _add_watermark(fig)
    out_path = os.path.join(output_dir, 'eis_rct_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Rct karşılaştırma grafiği: {out_path}")

# ══════════════════════════════════════════════════════════════
# 4. KOMPLE EIS ANALİZİ
# ══════════════════════════════════════════════════════════════

def run_eis_analysis(file_dict, output_dir='results_eis', sign_zimag=-1, plot_style=None):
    """
    Birden fazla elektrot/adım için tam EIS analizi.

    Parameters
    ----------
    file_dict : dict
        {'Bare GCE': 'data/eis_bare.csv',
         'GCE/rGO':  'data/eis_rgo.csv', ...}
    output_dir : str
    sign_zimag : int
        -1: Z_imag negatif olarak verilmiş (çoğu potansiyostat)

    Returns
    -------
    DataFrame: parametre tablosu
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'█'*55}")
    print("  EIS ANALİZİ BAŞLADI")
    print(f"{'█'*55}")

    fit_results = []
    for label, path in file_dict.items():
        print(f"\n  ► {label} işleniyor...")
        try:
            df = load_eis_file(path, sign_zimag=sign_zimag)
            print(f"    {len(df)} frekans noktası yüklendi")
            res = fit_eis(df, label=label, output_dir=output_dir)
            if res:
                fit_results.append(res)
                pe = res.get('perr', [np.nan]*5)
                print(f"    Rs    = {res['Rs']:.2f} ± {pe[0]:.2f} Ω")
                print(f"    Rct   = {res['Rct']:.2f} ± {pe[1]:.2f} Ω")
                print(f"    Q     = {res['Q']:.4e} ± {pe[2]:.2e} S·s^n")
                print(f"    n     = {res['n_cpe']:.4f} ± {pe[3]:.4f}")
                print(f"    σ     = {res['sigma']:.4f} ± {pe[4]:.4f} Ω·s^(-1/2)")
                print(f"    C_eff = {res['C_eff_F']:.4e} F")
                print(f"    χ²    = {res['chi2']:.4e}  {'✓ iyi fit' if res['chi2'] < 1e-3 else '⚠ fit kontrol edin'}")
        except Exception as e:
            print(f"    HATA: {e}")

    if not fit_results:
        print("HATA: Hiç veri işlenemedi.")
        return None

    # Nyquist karşılaştırma grafiği
    plot_nyquist_comparison(fit_results, output_dir=output_dir, plot_style=plot_style)

    # Bode grafikleri (EIS yayınlarında standart)
    plot_bode(fit_results, output_dir=output_dir, plot_style=plot_style)

    # Rct karşılaştırma
    plot_rct_comparison(fit_results, output_dir=output_dir, plot_style=plot_style)

    # Parametre tablosu
    rows = []
    for r in fit_results:
        pe = r.get('perr', [np.nan]*5)
        rows.append({
            'Electrode':      r['label'],
            'Rs (Ω)':         f"{r['Rs']:.2f}",
            'Rs_err (Ω)':     f"{pe[0]:.2f}",
            'Rct (Ω)':        f"{r['Rct']:.2f}",
            'Rct_err (Ω)':    f"{pe[1]:.2f}",
            'Q (S·sⁿ)':       f"{r['Q']:.4e}",
            'Q_err':          f"{pe[2]:.2e}",
            'n (CPE)':        f"{r['n_cpe']:.4f}",
            'n_err':          f"{pe[3]:.4f}",
            'σ (Ω·s⁻¹/²)':   f"{r['sigma']:.4f}",
            'σ_err':          f"{pe[4]:.4f}",
            'C_eff (F)':      f"{r['C_eff_F']:.4e}",
            'χ² (mod.wt.)':   f"{r['chi2']:.4e}",
        })
    table_df = pd.DataFrame(rows)

    table_path = os.path.join(output_dir, 'eis_parameters.csv')
    table_df.to_csv(table_path, index=False)

    print(f"\n{'═'*55}")
    print("EIS PARAMETRE TABLOSU")
    print("═"*55)
    print(table_df.to_string(index=False))
    print(f"\nTablo kaydedildi: {table_path}")

    # Pürüzlülük faktörü (RF)
    if len(fit_results) >= 2:
        c_bare = fit_results[0]['C_eff_F']
        print(f"\n{'─'*55}")
        print("PÜRÜZLÜLÜK FAKTÖRLERİ (bare elektrota göre)")
        print(f"{'─'*55}")
        for r in fit_results[1:]:
            rf = r['C_eff_F'] / c_bare
            print(f"  {r['label']}: RF = {rf:.3f}  "
                  f"(C_eff = {r['C_eff_F']:.4e} F)")

    return table_df, fit_results


# ══════════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Modifikasyon adımlarını sırayla ver
    file_dict = {
        'Bare GCE':     'data/eis_bare.csv',
        'GCE/Material': 'data/eis_step1.csv',
        'Modified GCE': 'data/eis_step2.csv',
    }

    results = run_eis_analysis(
        file_dict=file_dict,
        output_dir='results_eis',
        sign_zimag=-1   # Z_imag negatif olarak verilmişse -1
    )
