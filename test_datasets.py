"""
=============================================================
TEST VERİ SETLERİ — 6 Farklı Senaryo
=============================================================
Ankara University / FS3P Lab

Veri setleri:
  1. Urik Asit   — rGO/PEDOT modifiye GCE, pH 7.0 PBS
  2. Pb2+ agir metal — Bi-film kapli GCE, SWASV
  3. Glukoz (non-enzimatik) — NiCo2O4/GCE, alkalin ortam
  4. SANKO-1: Mn3O4/AB/SPCE — Sertralin Hidroklorur (SERH)
     Kaynak: Sanko ve ark., J. Electroanal. Chem. 2026
     doi:10.1016/j.jelechem.2026.xxxxx
  5. SANKO-2: KB+CoS2/SPCE — Sertralin (SER), cok genis aralik
     Kaynak: Sanko ve ark., Anal. Bioanal. Chem. 2026
     doi:10.1007/s00216-026-06670-4
  6. SANKO-3: Cu-GQD@ZIF8/PANI/PA/GCE — Ure enzimatik biyosensor
     Kaynak: Sanko ve ark., Anal. Methods 2024
     doi:10.1039/d4ay01397b
=============================================================
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from cv_module       import run_cv_analysis
from eis_module      import run_eis_analysis
from ph_module       import run_ph_analysis
from analytical_module import run_analytical_performance

np.random.seed(42)   # tekrarlanabilirlik


# ══════════════════════════════════════════════════════════════
# YARDIMCI: CV verisi üret (gerçekçi Randles-Ševčík)
# ══════════════════════════════════════════════════════════════

def make_cv_data(rate_mVs, ecsa_cm2=0.18, n=1, D=7.6e-6, C_mM=1.0,
                 E_half=0.22, dEp_base_mV=72, noise_frac=0.015,
                 n_pts=200):
    """
    Randles-Ševčík tabanlı sentetik CV eğrisi üretir.
    Gerçek CV şekline yakın, Gaussian pik profiliyle.
    """
    nu = rate_mVs / 1000  # V/s
    C  = C_mM * 1e-3 * 1e-3  # mol/cm³

    # Teorik pik akımı (Randles-Ševčík)
    ip_theory = 2.69e5 * (n**1.5) * ecsa_cm2 * np.sqrt(D) * C * np.sqrt(nu)

    # ΔEp hız bağımlılığı (kinetics → biraz artar)
    dEp_V = (dEp_base_mV + rate_mVs * 0.05) / 1000

    Epa = E_half + dEp_V / 2
    Epc = E_half - dEp_V / 2

    E = np.linspace(-0.1, 0.6, n_pts)

    # Anodik pik (Gaussian)
    sigma_a = 0.040 + rate_mVs * 2e-5
    I_anodic = ip_theory * np.exp(-0.5 * ((E - Epa) / sigma_a) ** 2)

    # Katodik pik (negatif)
    sigma_c = sigma_a * 1.1
    I_cathodic = -ip_theory * 0.95 * np.exp(-0.5 * ((E - Epc) / sigma_c) ** 2)

    # Kapasitif arka plan
    I_bg = 2e-7 * nu * (E - 0.25)

    I = I_anodic + I_cathodic + I_bg
    noise = np.random.normal(0, ip_theory * noise_frac, n_pts)
    I += noise

    return pd.DataFrame({'Potential': E, 'Current': I})


def make_eis_data(Rs=35, Rct=820, Q=8e-7, n_cpe=0.88, sigma=120,
                  n_pts=35, f_min=0.01, f_max=1e5):
    """
    Randles-CPE devre parametrelerinden sentetik EIS verisi üretir.
    """
    from eis_module import randles_cpe_impedance
    freq  = np.logspace(np.log10(f_min), np.log10(f_max), n_pts)
    omega = 2 * np.pi * freq
    Z     = randles_cpe_impedance(omega, Rs, Rct, Q, n_cpe, sigma)

    # %2 Gaussian gürültü
    noise_r = np.random.normal(0, np.abs(Z.real) * 0.015)
    noise_i = np.random.normal(0, np.abs(Z.imag) * 0.015)

    df = pd.DataFrame({
        'freq':   freq,
        'Zr':     Z.real + noise_r,
        'Zi':     Z.imag + noise_i   # negatif (standart potansiyostat çıkışı)
    })
    return df


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 1: ÜRİK ASİT (rGO/PEDOT/GCE)
# ══════════════════════════════════════════════════════════════

def prepare_dataset_uric_acid(base_dir='test_data/uric_acid'):
    """
    Ürik asit sensörü — DPV ile tayini, pH 7.0 PBS ortamı.
    Tipik sistem: rGO/PEDOT/GCE
    Literatür referansı: ACS Appl. Mater. Inter. tarzı yayınlar
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV hız taraması (Fe(CN)6 ile ECSA) ---
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}
    for rate in [10, 25, 50, 75, 100, 150, 200]:
        df = make_cv_data(rate, ecsa_cm2=0.213, dEp_base_mV=70, E_half=0.218)
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (bare → rGO → rGO/PEDOT) ---
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare GCE':       dict(Rs=38,  Rct=1850, Q=5.2e-7, n_cpe=0.90, sigma=145),
        'GCE/rGO':        dict(Rs=36,  Rct=620,  Q=1.1e-6, n_cpe=0.87, sigma=95),
        'GCE/rGO/PEDOT':  dict(Rs=34,  Rct=215,  Q=2.3e-6, n_cpe=0.85, sigma=72),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = label.replace('/', '_').replace(' ', '_').lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- pH taraması ---
    ph_data = {
        'pH': [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5],
        # Ep vs pH eğimi: −57.1 mV/pH (UA: 2H+/2e-)
        'Ep_V': [0.385, 0.356, 0.327, 0.298, 0.269, 0.248, 0.212, 0.183],
        # ip maksimum pH 7.0'da
        'ip_A': [1.82e-6, 2.14e-6, 2.78e-6, 3.42e-6, 3.95e-6,
                 3.71e-6, 3.12e-6, 2.68e-6]
    }
    ph_path = os.path.join(base_dir, 'ph_data.csv')
    pd.DataFrame(ph_data).to_csv(ph_path, index=False)

    # --- Kalibrasyon (DPV, 0.1–200 μM) ---
    slope_true  = 0.0582   # μA/μM
    intercept_true = 0.021
    concs  = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    cal_data = {}
    for c in concs:
        ip_true = slope_true * c + intercept_true
        cal_data[c] = [ip_true * (1 + np.random.normal(0, 0.012)) for _ in range(3)]

    # --- RSD ---
    rsd_data = {
        'Intra-day': {'n=1': [3.952, 3.971, 3.948, 3.963, 3.956]},  # 10 μM
        'Inter-day': {'Day 1': [3.952, 3.971],
                      'Day 2': [3.948, 3.963],
                      'Day 3': [3.962, 3.945]}
    }

    # --- Geri kazanım ---
    recovery_data = {
        'Matriksler': ['Serum', 'İdrar', 'Çeşme suyu'],
        'spike_uM':   [10.0, 10.0, 10.0],
        'ip_uA': [
            [0.6007, 0.5994, 0.6018],
            [0.5981, 0.6025, 0.5998],
            [0.5991, 0.6008, 0.5987],
        ]
    }

    # --- Standart ilave ---
    # Numunedeki UA konsantrasyonu: 3.57 μM (örn. seyreltilmiş serum)
    C_sample_true = 3.57
    std_add_additions = [0, 2, 4, 6, 8, 10]
    std_add_ip = [slope_true * (C_sample_true + c) + intercept_true +
                  np.random.normal(0, 0.015) for c in std_add_additions]

    return {
        'cv_files': cv_files,
        'eis_files': eis_files,
        'ph_path': ph_path,
        'cal_data': cal_data,
        'rsd_data': rsd_data,
        'recovery_data': recovery_data,
        'std_addition': {
            'additions_uM': std_add_additions,
            'ip_uA': std_add_ip
        }
    }


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 2: Pb²⁺ AĞIR METAL (Bi-film/GCE, SWASV)
# ══════════════════════════════════════════════════════════════

def prepare_dataset_lead(base_dir='test_data/lead_swasv'):
    """
    Kurşun (Pb²⁺) tespiti — Bismuth film kaplanmış GCE, SWASV
    Lineer aralık: 0.5–50 μg/L (ppb)
    Tipik sistem: Bi-film/GCE
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV (Bi-film elektrotta Fe(CN)6 ile karakterizasyon) ---
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}
    for rate in [10, 25, 50, 100, 150, 200]:
        df = make_cv_data(rate, ecsa_cm2=0.195, dEp_base_mV=85,
                          E_half=0.215, noise_frac=0.02)
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (bare → Bi-film) ---
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare GCE':     dict(Rs=42, Rct=2200, Q=4.8e-7, n_cpe=0.89, sigma=165),
        'GCE/Bi-film':  dict(Rs=39, Rct=890,  Q=9.5e-7, n_cpe=0.86, sigma=118),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = label.replace('/', '_').replace(' ', '_').lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- Kalibrasyon (SWASV, 0.5–50 μg/L) ---
    # Pb²⁺ SWASV — stripping peak current
    slope_true  = 0.412   # μA per μg/L
    intercept_true = 0.038
    concs  = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    cal_data = {}
    for c in concs:
        ip_true = slope_true * c + intercept_true
        cal_data[c] = [ip_true * (1 + np.random.normal(0, 0.018)) for _ in range(3)]

    # --- RSD (5 μg/L) ---
    ip_5 = slope_true * 5.0 + intercept_true
    rsd_data = {
        'Intra-day': {'n=1': [ip_5 * (1 + np.random.normal(0, 0.01)) for _ in range(5)]},
        'Inter-day': {
            f'Day {i}': [ip_5 * (1 + np.random.normal(0, 0.015)) for _ in range(2)]
            for i in range(1, 4)
        }
    }

    # --- Geri kazanım ---
    recovery_data = {
        'Matriksler': ['Nehir suyu', 'Musluk suyu', 'Yağmur suyu'],
        'spike_uM':   [5.0, 5.0, 5.0],
        'ip_uA': [
            [ip_5 * (1 + np.random.normal(0, 0.01)) for _ in range(3)],
            [ip_5 * (1 + np.random.normal(0, 0.012)) for _ in range(3)],
            [ip_5 * (1 + np.random.normal(0, 0.009)) for _ in range(3)],
        ]
    }

    return {
        'cv_files': cv_files,
        'eis_files': eis_files,
        'ph_path': None,
        'cal_data': cal_data,
        'rsd_data': rsd_data,
        'recovery_data': recovery_data,
        'std_addition': None   # SWASV için standart ilave genelde kalibrasyon yerine kullanılır
    }


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 3: GLUKOZ — NiCo₂O₄/GCE (non-enzimatik)
# ══════════════════════════════════════════════════════════════

def prepare_dataset_glucose(base_dir='test_data/glucose'):
    """
    Non-enzimatik glukoz tespiti — NiCo₂O₄/GCE, 0.1 M NaOH
    Lineer aralık: 5–1500 μM (geniş)
    Adsorpsiyon kontrollü mekanizma (b ≈ 0.95)
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV (adsorpsiyon kontrollü: ip ∝ ν, b≈0.95) ---
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}

    for rate in [10, 25, 50, 75, 100, 150, 200]:
        # Adsorpsiyon kontrollü: ip ∝ ν^0.95 (yaklaşık b=0.95)
        nu   = rate / 1000
        ecsa = 0.0   # ip direkt parametrik
        # ip = k × ν^0.95 yaklaşık olarak
        ip_base = 1.82e-5 * (nu**0.95)
        sigma_a = 0.055

        E = np.linspace(-0.1, 0.65, 200)
        E_half = 0.42  # NiCo elektrot
        dEp = (58 + rate * 0.03) / 1000

        Epa = E_half + dEp / 2
        Epc = E_half - dEp / 2

        I = (ip_base * np.exp(-0.5 * ((E - Epa) / sigma_a) ** 2)
             - ip_base * 0.90 * np.exp(-0.5 * ((E - Epc) / (sigma_a * 1.1)) ** 2)
             + 3e-7 * nu * (E - 0.3))
        I += np.random.normal(0, ip_base * 0.015, len(E))

        df = pd.DataFrame({'Potential': E, 'Current': I})
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (bare → NiCo₂O₄) ---
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare GCE':         dict(Rs=28, Rct=3100, Q=3.5e-7, n_cpe=0.91, sigma=185),
        'GCE/NiCo₂O₄':     dict(Rs=25, Rct=185,  Q=4.2e-6, n_cpe=0.84, sigma=55),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = ''.join(c if c.isalnum() or c == '_' else '_' for c in label).lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- pH (NaOH ortamı: 11–14 arası bak) ---
    ph_data = {
        'pH': [11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
        'Ep_V': [0.455, 0.448, 0.439, 0.430, 0.421, 0.415],
        'ip_A': [2.1e-6, 2.95e-6, 3.78e-6, 4.42e-6, 4.85e-6, 4.72e-6]
    }
    ph_path = os.path.join(base_dir, 'ph_data.csv')
    pd.DataFrame(ph_data).to_csv(ph_path, index=False)

    # --- Kalibrasyon (geniş lineer aralık: 5–1500 μM) ---
    slope_true  = 0.00318   # μA/μM
    intercept_true = 0.011
    concs  = [5, 10, 25, 50, 100, 200, 500, 1000, 1500]
    cal_data = {}
    for c in concs:
        ip_true = slope_true * c + intercept_true
        cal_data[c] = [ip_true * (1 + np.random.normal(0, 0.016)) for _ in range(3)]

    # --- RSD (100 μM glukoz) ---
    ip_100 = slope_true * 100 + intercept_true
    rsd_data = {
        'Intra-day': {'n=1': [ip_100 * (1 + np.random.normal(0, 0.01)) for _ in range(5)]},
        'Inter-day': {
            f'Day {i}': [ip_100 * (1 + np.random.normal(0, 0.014)) for _ in range(2)]
            for i in range(1, 4)
        }
    }

    # --- Geri kazanım ---
    recovery_data = {
        'Matriksler': ['İnsan serumu', 'İdrar', 'Sporcu içeceği'],
        'spike_uM':   [100.0, 100.0, 100.0],
        'ip_uA': [
            [ip_100 * (1 + np.random.normal(0, 0.012)) for _ in range(3)],
            [ip_100 * (1 + np.random.normal(0, 0.015)) for _ in range(3)],
            [ip_100 * (1 + np.random.normal(0, 0.010)) for _ in range(3)],
        ]
    }

    # --- Standart ilave (gerçek numune: ~82 μM) ---
    C_sample_true = 82.0
    std_add_additions = [0, 25, 50, 75, 100]
    std_add_ip = [slope_true * (C_sample_true + c) + intercept_true +
                  np.random.normal(0, 0.008) for c in std_add_additions]

    return {
        'cv_files': cv_files,
        'eis_files': eis_files,
        'ph_path': ph_path,
        'cal_data': cal_data,
        'rsd_data': rsd_data,
        'recovery_data': recovery_data,
        'std_addition': {
            'additions_uM': std_add_additions,
            'ip_uA': std_add_ip
        }
    }


# ══════════════════════════════════════════════════════════════
# YARDIMCI: Karisik mekanizma CV verisi (ip oc v^b)
# ══════════════════════════════════════════════════════════════

def make_cv_mixed(rate_mVs, b=0.735, ip_ref=4.5e-6, E_half=0.84,
                  dEp_base_mV=85, noise_frac=0.015, n_pts=200,
                  rate_ref=100, E_start=0.3, E_end=1.4, peak_type='anodic'):
    """
    ip oc v^b ile CV verisi uretir (karisik difuzyon-adsorpsiyon).
    b=0.5: saf difuzyon (Randles-Sevcik)
    b=1.0: saf adsorpsiyon
    """
    nu     = rate_mVs / 1000
    nu_ref = rate_ref  / 1000
    ip_theory = ip_ref * (nu / nu_ref)**b

    dEp_V = (dEp_base_mV + rate_mVs * 0.04) / 1000
    if peak_type == 'cathodic':
        Epa = E_half - dEp_V / 2
        Epc = E_half + dEp_V / 2  # cathodic is more negative
    else:
        Epa = E_half + dEp_V / 2
        Epc = E_half - dEp_V / 2

    E = np.linspace(E_start, E_end, n_pts)
    sigma_a = 0.045 + rate_mVs * 1.5e-5

    if peak_type == 'cathodic':
        I = (-ip_theory * np.exp(-0.5 * ((E - Epa) / sigma_a) ** 2)
             + ip_theory * 0.90 * np.exp(-0.5 * ((E - Epc) / (sigma_a * 1.1)) ** 2)
             + 2e-7 * nu * (E - E_half))
    else:
        I = (ip_theory * np.exp(-0.5 * ((E - Epa) / sigma_a) ** 2)
             - ip_theory * 0.90 * np.exp(-0.5 * ((E - Epc) / (sigma_a * 1.1)) ** 2)
             + 2e-7 * nu * (E - E_half))

    I += np.random.normal(0, ip_theory * noise_frac, n_pts)
    return pd.DataFrame({'Potential': E, 'Current': I})


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 4: SANKO-1
# Mn3O4/AB/SPCE — Sertralin Hidroklorur (SERH) tespiti
# Kaynak: Sanko ve ark., J. Electroanal. Chem. 2026
# Parametreler dogrudan yayindan alinmistir.
# ══════════════════════════════════════════════════════════════

def prepare_dataset_sanko(base_dir='test_data/sanko_1_mn3o4_serh'):
    """
    SANKO-1: Mn3O4/AB/SPCE — Sertralin Hidroklorur (SERH) tespiti
    Kaynak: Sanko ve ark., J. Electroanal. Chem. 2026
    doi:10.1016/j.jelechem.2026.xxxxx

    Yayindan alinan parametreler:
      - b (log ip vs log v egimi) = 0.7352, R2 = 0.9905
      - Ep vs log v egimi = 0.0896 V/dekad -> a*n = 0.66, n=2, a=0.33
      - pH egimi: Ep(V) = -0.061*pH + 1.435, R2 = 0.9869; optimal pH 7.4
      - EIS Rct: Yalın SPCE=4350, AB/SPCE=3544, Mn3O4/SPCE=3394, Mn3O4/AB/SPCE=3183 Ohm
      - DPV kalibrasyon: ip(uA) = 0.0251*C(uM) + 6.574, R2=0.9982
      - Aralik: 50 nM - 1.0 uM; LOD = 18 nM
      - ECSA (Mn3O4/AB/SPCE) = 0.0177 cm2
      - Geri kazanim (serum): 99.66-100.59%
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV hiz taramasi (karisik mekanizma, b=0.7352) ---
    # ip_ref: 100 mV/s'de ~6 uA (SPCE kucuk yuzey, dusuk ECSA=0.0177 cm2)
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}
    # Yayindaki tarama hizlari: 25, 50, 100, 150, 200, 300, 350, 400 mV/s
    for rate in [25, 50, 100, 150, 200, 300, 350, 400]:
        df = make_cv_mixed(rate, b=0.7352, ip_ref=6.2e-6, E_half=0.884,
                           dEp_base_mV=89, noise_frac=0.012,
                           rate_ref=100, E_start=0.4, E_end=1.3)
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (4 elektrot): Yalın SPCE → AB/SPCE → Mn3O4/SPCE → Mn3O4/AB/SPCE ---
    # Rct yayindan; Q ve n_cpe SPCE elektrota uygun tahmin
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare SPCE':          dict(Rs=38, Rct=4350, Q=3.8e-7, n_cpe=0.87, sigma=210),
        'AB/SPCE':            dict(Rs=36, Rct=3544, Q=5.2e-7, n_cpe=0.86, sigma=175),
        'Mn3O4/SPCE':         dict(Rs=35, Rct=3394, Q=6.1e-7, n_cpe=0.85, sigma=162),
        'Mn3O4/AB/SPCE':      dict(Rs=34, Rct=3183, Q=7.8e-7, n_cpe=0.84, sigma=148),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = label.replace('/', '_').replace(' ', '_').lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- pH taramasi (SERH: 2H+/2e-) ---
    # Ep(V) = -0.061*pH + 1.435 (yayindan), optimal pH 7.4
    # pH aralik: 5.0 - 9.0 (tampon: PBS/B-R)
    pH_vals = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5, 9.0]
    Ep_vals  = [-0.061 * p + 1.435 + np.random.normal(0, 0.003) for p in pH_vals]
    # ip maksimum pH 7.4'te (yayindan); Gaussian profil
    ip_center = 7.4
    ip_max    = 8.5e-6
    ip_vals   = [ip_max * np.exp(-0.5 * ((p - ip_center) / 1.0)**2)
                 * (1 + np.random.normal(0, 0.02)) for p in pH_vals]
    ph_path = os.path.join(base_dir, 'ph_data.csv')
    pd.DataFrame({'pH': pH_vals, 'Ep_V': Ep_vals, 'ip_A': ip_vals}).to_csv(ph_path, index=False)

    # --- DPV kalibrasyon (nM aralik; 50 nM - 1000 nM = 1 uM) ---
    # Yayindan: ip(uA) = 0.0251*C(uM) + 6.574, R2=0.9982, LOD=18 nM
    # Konsantrasyonlar uM cinsinden
    slope_true     = 0.0251   # uA/uM
    intercept_true = 6.574    # uA (arka plan DPV akimi dahil)
    concs_uM = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    cal_data = {}
    for c in concs_uM:
        ip_true = slope_true * c + intercept_true
        # R2=0.9982 icin dusuk gurultu
        cal_data[c] = [ip_true * (1 + np.random.normal(0, 0.004)) for _ in range(3)]

    # --- RSD (0.5 uM SERH, intra/inter) ---
    ip_ref_rsd = slope_true * 0.5 + intercept_true
    rsd_data = {
        'Intra-day (n=5)': {'Tekrar': [ip_ref_rsd * (1 + np.random.normal(0, 0.006)) for _ in range(5)]},
        'Inter-day':       {f'Gun {i}': [ip_ref_rsd * (1 + np.random.normal(0, 0.009)) for _ in range(3)]
                            for i in range(1, 4)}
    }

    # --- Geri kazanim (insan serumu, 3 konsantrasyon) ---
    # Yayindan: 99.66-100.59% (serum); spike: 0.2, 0.5, 0.8 uM
    recovery_data = {
        'Matriksler': ['Serum (0.2 uM)', 'Serum (0.5 uM)', 'Serum (0.8 uM)'],
        'spike_uM':   [0.2, 0.5, 0.8],
        'ip_uA': [
            [(slope_true*0.2 + intercept_true) * (1 + np.random.normal(0, 0.003)) for _ in range(3)],
            [(slope_true*0.5 + intercept_true) * (1 + np.random.normal(0, 0.003)) for _ in range(3)],
            [(slope_true*0.8 + intercept_true) * (1 + np.random.normal(0, 0.003)) for _ in range(3)],
        ]
    }

    # --- Standart ilave (numune: ~0.35 uM SERH) ---
    C_sample_true = 0.35
    std_add_additions = [0, 0.2, 0.4, 0.6, 0.8]
    std_add_ip = [slope_true * (C_sample_true + c) + intercept_true +
                  np.random.normal(0, 0.005) for c in std_add_additions]

    return {
        'cv_files':      cv_files,
        'eis_files':     eis_files,
        'ph_path':       ph_path,
        'cal_data':      cal_data,
        'rsd_data':      rsd_data,
        'recovery_data': recovery_data,
        'std_addition':  {'additions_uM': std_add_additions, 'ip_uA': std_add_ip}
    }


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 5: SANKO-2
# KB+CoS2/SPCE — Sertralin (SER) tespiti, log-lineer aralik
# Kaynak: Sanko ve ark., Anal. Bioanal. Chem. 2026
# doi:10.1007/s00216-026-06670-4
# ══════════════════════════════════════════════════════════════

def prepare_dataset_sanko2(base_dir='test_data/sanko_2_cob_ser'):
    """
    SANKO-2: KB+CoS2/SPCE — Sertralin (SER) tespiti

    Yayindan alinan parametreler:
      - b = 0.806; karisik mekanizma (ag. difuzyon + adsorpsiyon)
      - Scan rates: 5-150 mV/s
      - pH egimi: Epa(V) = -0.0536*pH + 0.949, R2=0.9833; optimal pH 8.0; 2H+/2e-
      - EIS Rct: Yalin SPCE=4440, KB/SPCE=3540, CoS2/SPCE=103, KB+CoS2/SPCE=83.1 Ohm
        CPE N ~ 0.3-0.5 (cok puruzlu yuzey)
      - DPV log-lineer: ip(uA) = 16.881*log10(C/pM) + 18.453, R2=0.9774
        Aralik: 1 nM - 1 uM; LOD = 0.0037 nM
      - %RSD = 1.32%; Geri kazanim: yuzey suyu %108-140, atik su %103-151

    NOT: Kalibrasyon ekseni log10(C_nM) olarak ifade edilmistir
    (modul lineer regresyon yapar; x = log10(C/nM) girilmeli).
    ip(uA) = 16.881*x + 69.096  burada x = log10(C/nM)
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV (karisik mekanizma, b=0.806) ---
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}
    for rate in [5, 10, 25, 50, 75, 100, 150]:
        df = make_cv_mixed(rate, b=0.806, ip_ref=5.8e-6, E_half=0.520,
                           dEp_base_mV=82, noise_frac=0.015,
                           rate_ref=50, E_start=0.1, E_end=0.95)
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (4 elektrot): Yalin → KB/SPCE → CoS2/SPCE → KB+CoS2/SPCE ---
    # Rct degerlerinden CoS2 cok iletken (103 Ohm); CPE n=0.40 (pürüzlü)
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare SPCE':      dict(Rs=42, Rct=4440, Q=3.5e-7, n_cpe=0.82, sigma=220),
        'KB/SPCE':        dict(Rs=40, Rct=3540, Q=5.8e-7, n_cpe=0.78, sigma=180),
        'CoS2/SPCE':      dict(Rs=38, Rct=103,  Q=4.2e-6, n_cpe=0.42, sigma=55),
        'KB+CoS2/SPCE':   dict(Rs=37, Rct=83,   Q=5.1e-6, n_cpe=0.40, sigma=48),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = label.replace('/', '_').replace(' ', '_').replace('+', 'p').lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- pH taramasi (SER: 2H+/2e-) ---
    # Epa(V) = -0.0536*pH + 0.949 (yayindan), optimal pH 8.0
    pH_vals = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5, 9.0]
    Ep_vals  = [-0.0536 * p + 0.949 + np.random.normal(0, 0.003) for p in pH_vals]
    ip_center = 8.0
    ip_max    = 9.2e-6
    ip_vals   = [ip_max * np.exp(-0.5 * ((p - ip_center) / 1.1)**2)
                 * (1 + np.random.normal(0, 0.02)) for p in pH_vals]
    ph_path = os.path.join(base_dir, 'ph_data.csv')
    pd.DataFrame({'pH': pH_vals, 'Ep_V': Ep_vals, 'ip_A': ip_vals}).to_csv(ph_path, index=False)

    # --- DPV log-lineer kalibrasyon ---
    # ip(uA) = 16.881 * log10(C_nM) + 69.096
    # Aralik: 1 nM (log=0) - 1000 nM (log=3)
    # x ekseni = log10(C/nM); 7 nokta
    slope_log     = 16.881   # uA per log decade
    intercept_log = 69.096   # uA at log(1 nM) = 0
    log_concs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # log10(C/nM)
    cal_data = {}
    for lc in log_concs:
        ip_true = slope_log * lc + intercept_log
        # R2=0.9774 -> orta gurultu
        cal_data[lc] = [ip_true * (1 + np.random.normal(0, 0.012)) for _ in range(3)]

    # --- RSD (100 nM -> log=2) ---
    ip_ref_rsd = slope_log * 2.0 + intercept_log
    rsd_data = {
        'Intra-day (n=5)': {'Tekrar': [ip_ref_rsd * (1 + np.random.normal(0, 0.0066)) for _ in range(5)]},
        'Inter-day':       {f'Gun {i}': [ip_ref_rsd * (1 + np.random.normal(0, 0.010)) for _ in range(3)]
                            for i in range(1, 4)}
    }

    # --- Geri kazanim (cevresel numuneler) ---
    # Yayindan: yuzey suyu %108-131, atiksu %103-151
    ip_spike = slope_log * 2.0 + intercept_log  # 100 nM spike icin
    recovery_data = {
        'Matriksler': ['Yuzey suyu', 'Atiksu'],
        'spike_uM':   [0.1, 0.1],  # 100 nM = 0.1 uM
        'ip_uA': [
            [ip_spike * 1.19, ip_spike * 1.14, ip_spike * 1.08],  # yuzey suyu %108-120
            [ip_spike * 1.03, ip_spike * 1.08, ip_spike * 1.12],  # atiksu %103-112
        ]
    }

    # --- Standart ilave (log uzayda; numune ~log(10 nM)=1.0 birim) ---
    C_sample_log = 1.0   # 10 nM numune
    std_add_log  = [0, 0.3, 0.6, 0.9, 1.2, 1.5]  # eklenen log birimleri
    std_add_ip   = [slope_log * (C_sample_log + s) + intercept_log +
                    np.random.normal(0, 0.5) for s in std_add_log]

    return {
        'cv_files':      cv_files,
        'eis_files':     eis_files,
        'ph_path':       ph_path,
        'cal_data':      cal_data,
        'rsd_data':      rsd_data,
        'recovery_data': recovery_data,
        'std_addition':  {'additions_uM': std_add_log, 'ip_uA': std_add_ip}
    }


# ══════════════════════════════════════════════════════════════
# VERİ SETİ 6: SANKO-3
# Cu-GQD@ZIF8/PANI/PA/GCE — Ure enzimatik biyosensor
# Kaynak: Sanko ve ark., Anal. Methods 2024
# doi:10.1039/d4ay01397b
# ══════════════════════════════════════════════════════════════

def prepare_dataset_sanko3(base_dir='test_data/sanko_3_urea'):
    """
    SANKO-3: Cu-GQD@ZIF8/PANI/PA/GCE — Ure enzimatik biyosensor

    Yayindan alinan parametreler:
      - DPV kalibrasyon: ip(uA) = 0.0323*C(uM) + 0.1664, R2=0.9913
      - Aralik: 1 - 80 uM ure; LOD = 0.77 uM, LOQ = 2.31 uM
      - pH 7.0 PBS ortami; urease enzimi katodik amonyak piki
      - Gercek numune: yapay serum

    NOT: Pik katodik (amonyak oksidasyonu/indirgemesi); ip pozitif
    olarak tanimlanmistir (DPV peak current magnitude).
    EIS verileri literatur benzerine gore tahmin edilmistir
    (yayinda sayisal deger verilmemistir).
    """
    os.makedirs(base_dir, exist_ok=True)

    # --- CV hiz taramasi (urease elektrodu, karisik/adsorpsiyon, b~0.68) ---
    cv_dir = os.path.join(base_dir, 'cv_data')
    os.makedirs(cv_dir, exist_ok=True)
    cv_files = {}
    # Katodik pik (amonyak): E_half ~ -0.35 V vs Ag/AgCl (0.1 M PBS)
    for rate in [10, 25, 50, 75, 100, 150, 200]:
        df = make_cv_mixed(rate, b=0.680, ip_ref=4.8e-6, E_half=-0.35,
                           dEp_base_mV=72, noise_frac=0.015,
                           rate_ref=50, E_start=-0.8, E_end=0.2,
                           peak_type='cathodic')
        path = os.path.join(cv_dir, f'cv_{rate}mVs.csv')
        df.to_csv(path, index=False)
        cv_files[rate] = path

    # --- EIS (5 katman: GCE → PANI → PA/PANI → ZIF8@GCE → Cu-GQD@ZIF8/PANI/PA) ---
    # Tahmin: her katman Rct'yi azaltiyor (Cu-GQD cok iletken)
    eis_dir = os.path.join(base_dir, 'eis_data')
    os.makedirs(eis_dir, exist_ok=True)
    eis_configs = {
        'Bare GCE':                   dict(Rs=28, Rct=2850, Q=4.2e-7, n_cpe=0.90, sigma=190),
        'GCE/PANI':                   dict(Rs=26, Rct=1240, Q=8.5e-7, n_cpe=0.87, sigma=130),
        'GCE/PA/PANI':                dict(Rs=25, Rct=780,  Q=1.6e-6, n_cpe=0.85, sigma=98),
        'GCE/ZIF8':                   dict(Rs=24, Rct=420,  Q=3.1e-6, n_cpe=0.84, sigma=72),
        'Cu-GQD@ZIF8/PANI/PA/GCE':   dict(Rs=23, Rct=185,  Q=5.4e-6, n_cpe=0.83, sigma=52),
    }
    eis_files = {}
    for label, params in eis_configs.items():
        df = make_eis_data(**params)
        fname = ''.join(c if c.isalnum() or c == '_' else '_' for c in label).lower() + '.csv'
        path  = os.path.join(eis_dir, fname)
        df.to_csv(path, index=False)
        eis_files[label] = path

    # --- pH: PBS pH 7.0 secilmis (tarama: 5.5-8.0); optimal pH 7.0 ---
    pH_vals = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    # Katodik Ep vs pH: yakin Nernst (-59 mV/pH, 1H+/1e- icin)
    Ep_base_pH7 = -0.350
    Ep_vals  = [Ep_base_pH7 - 0.059 * (p - 7.0) + np.random.normal(0, 0.003) for p in pH_vals]
    ip_center = 7.0
    ip_max    = 6.5e-6
    ip_vals   = [ip_max * np.exp(-0.5 * ((p - ip_center) / 0.8)**2)
                 * (1 + np.random.normal(0, 0.02)) for p in pH_vals]
    ph_path = os.path.join(base_dir, 'ph_data.csv')
    pd.DataFrame({'pH': pH_vals, 'Ep_V': Ep_vals, 'ip_A': ip_vals}).to_csv(ph_path, index=False)

    # --- DPV kalibrasyon (1-80 uM ure) ---
    # Yayindan: ip(uA) = 0.0323*C + 0.1664, R2=0.9913; LOD=0.77 uM
    slope_true     = 0.0323   # uA/uM
    intercept_true = 0.1664   # uA
    concs_uM = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 80.0]
    cal_data = {}
    for c in concs_uM:
        ip_true = slope_true * c + intercept_true
        # R2=0.9913 -> orta gurultu
        cal_data[c] = [ip_true * (1 + np.random.normal(0, 0.010)) for _ in range(3)]

    # --- RSD (20 uM ure) ---
    ip_ref_rsd = slope_true * 20.0 + intercept_true
    rsd_data = {
        'Intra-day (n=5)': {'Tekrar': [ip_ref_rsd * (1 + np.random.normal(0, 0.010)) for _ in range(5)]},
        'Inter-day':       {f'Gun {i}': [ip_ref_rsd * (1 + np.random.normal(0, 0.014)) for _ in range(3)]
                            for i in range(1, 4)}
    }

    # --- Geri kazanim (yapay serum: 2 spike seviyesi) ---
    recovery_data = {
        'Matriksler': ['Yapay serum (10 uM)', 'Yapay serum (40 uM)'],
        'spike_uM':   [10.0, 40.0],
        'ip_uA': [
            [(slope_true*10.0 + intercept_true) * (1 + np.random.normal(0, 0.010)) for _ in range(3)],
            [(slope_true*40.0 + intercept_true) * (1 + np.random.normal(0, 0.010)) for _ in range(3)],
        ]
    }

    # --- Standart ilave (numune: ~15 uM ure) ---
    C_sample_true = 15.0
    std_add_additions = [0, 10, 20, 30, 40]
    std_add_ip = [slope_true * (C_sample_true + c) + intercept_true +
                  np.random.normal(0, 0.015) for c in std_add_additions]

    return {
        'cv_files':      cv_files,
        'eis_files':     eis_files,
        'ph_path':       ph_path,
        'cal_data':      cal_data,
        'rsd_data':      rsd_data,
        'recovery_data': recovery_data,
        'std_addition':  {'additions_uM': std_add_additions, 'ip_uA': std_add_ip}
    }


# ══════════════════════════════════════════════════════════════
# TEST ÇALIŞTIRICISI
# ══════════════════════════════════════════════════════════════

def run_test(dataset_name, ds, out_root, analyte, unit='μM',
             area=0.071, n_el=2, diff_coeff=7.6e-6, conc_mM=1.0,
             ph_n_electrons=2):
    """Bir veri setini tüm modüllerle çalıştır."""
    print(f"\n{'▓'*65}")
    print(f"  TEST: {dataset_name}")
    print(f"{'▓'*65}")

    out = os.path.join(out_root, dataset_name.lower().replace(' ', '_'))
    os.makedirs(out, exist_ok=True)

    errors = []

    # 1. CV
    try:
        cv_out = os.path.join(out, 'cv')
        run_cv_analysis(
            file_dict=ds['cv_files'],
            concentration_mM=conc_mM,
            diffusion_coeff=diff_coeff,
            n_electrons=n_el,
            output_dir=cv_out
        )
        print(f"  ✓ CV analizi tamamlandı")
    except Exception as e:
        errors.append(f'CV: {e}')
        print(f"  ✗ CV HATA: {e}")

    # 2. EIS
    try:
        eis_out = os.path.join(out, 'eis')
        run_eis_analysis(
            file_dict=ds['eis_files'],
            output_dir=eis_out,
            sign_zimag=-1   # Zi negatif kaydedilmiş (standart potansiyostat çıkışı)
        )
        print(f"  ✓ EIS analizi tamamlandı")
    except Exception as e:
        errors.append(f'EIS: {e}')
        print(f"  ✗ EIS HATA: {e}")

    # 3. pH
    if ds.get('ph_path'):
        try:
            ph_out = os.path.join(out, 'ph')
            run_ph_analysis(
                source=ds['ph_path'],
                n_electrons=ph_n_electrons,
                output_dir=ph_out
            )
            print(f"  ✓ pH analizi tamamlandı")
        except Exception as e:
            errors.append(f'pH: {e}')
            print(f"  ✗ pH HATA: {e}")

    # 4. Analitik performans
    try:
        anal_out = os.path.join(out, 'analytical')
        run_analytical_performance(
            calibration_source=ds['cal_data'],
            analyte_name=analyte,
            electrode_area_cm2=area,
            unit=unit,
            rsd_data=ds.get('rsd_data'),
            recovery_data=ds.get('recovery_data'),
            std_addition_data=ds.get('std_addition'),
            output_dir=anal_out
        )
        print(f"  ✓ Analitik performans tamamlandı")
    except Exception as e:
        errors.append(f'Analytical: {e}')
        print(f"  ✗ Analitik HATA: {e}")

    if errors:
        print(f"\n  ⚠ HATALAR: {errors}")
    else:
        print(f"\n  ✅ Tüm modüller başarıyla çalıştı → {out}/")

    return errors


# ══════════════════════════════════════════════════════════════
# ANA ÇALIŞTIRICI
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_root = 'test_results'
    os.makedirs(out_root, exist_ok=True)

    all_errors = {}

    # ── Veri Seti 1: Ürik Asit ───────────────────────────────
    ds1 = prepare_dataset_uric_acid()
    all_errors['Uric Acid'] = run_test(
        'Uric_Acid', ds1, out_root,
        analyte='Uric Acid', unit='μM', area=0.071,
        n_el=1, diff_coeff=7.6e-6, ph_n_electrons=2
    )

    # ── Veri Seti 2: Pb²⁺ ───────────────────────────────────
    ds2 = prepare_dataset_lead()
    all_errors['Lead'] = run_test(
        'Lead_Pb2plus', ds2, out_root,
        analyte='Pb²⁺', unit='μg/L', area=0.071,
        n_el=1, diff_coeff=9.45e-6, ph_n_electrons=2
    )

    # ── Veri Seti 3: Glukoz ──────────────────────────────────
    ds3 = prepare_dataset_glucose()
    all_errors['Glucose'] = run_test(
        'Glucose', ds3, out_root,
        analyte='Glucose', unit='μM', area=0.071,
        n_el=2, diff_coeff=6.7e-6, ph_n_electrons=2
    )

    # ── Veri Seti 4: Şanko-1 / Mn3O4/AB/SPCE — SERH ─────────
    ds4 = prepare_dataset_sanko()
    all_errors['Sanko-1 (Mn3O4/AB/SPCE — SERH)'] = run_test(
        'Sanko1_Mn3O4_SERH', ds4, out_root,
        analyte='Sertraline HCl', unit='μM', area=0.0177,
        n_el=2, diff_coeff=5.8e-6, ph_n_electrons=2
    )

    # ── Veri Seti 5: Şanko-2 / KB+CoS2/SPCE — SER ───────────
    ds5 = prepare_dataset_sanko2()
    all_errors['Sanko-2 (KB+CoS2/SPCE — SER)'] = run_test(
        'Sanko2_CoS2_SER', ds5, out_root,
        analyte='Sertraline (log scale)', unit='log(nM)', area=0.0177,
        n_el=2, diff_coeff=5.8e-6, ph_n_electrons=2
    )

    # ── Veri Seti 6: Şanko-3 / Cu-GQD@ZIF8 — Üre ───────────
    ds6 = prepare_dataset_sanko3()
    all_errors['Sanko-3 (Cu-GQD@ZIF8 — Urea)'] = run_test(
        'Sanko3_CuGQD_Urea', ds6, out_root,
        analyte='Urea', unit='μM', area=0.0707,
        n_el=2, diff_coeff=1.38e-5, ph_n_electrons=1
    )

    # Özet
    print(f"\n{'═'*65}")
    print("TEST ÖZET")
    print("═"*65)
    for name, errs in all_errors.items():
        status = '✅' if not errs else f'⚠ {len(errs)} hata'
        print(f"  {name:<35}: {status}")
    print()
