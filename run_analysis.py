"""
=============================================================
ELEKTROKİMYASAL SENSÖR HESAPLAMA ARACI — ANA ÇALIŞTIRICI
=============================================================
Ankara University / FS3P Lab

KULLANIM:
  python run_analysis.py

Veya modülleri ayrı ayrı çalıştır:
  python cv_module.py
  python eis_module.py
  python ph_module.py
  python analytical_module.py

GEREKSİNİMLER (bir kez kur):
  pip install numpy pandas matplotlib scipy

GİRİŞ CSV FORMATLARI:
  Tüm CSV dosyaları için genel format: ilk satır başlık,
  sayısal veriler, ondalık ayırıcı nokta (.), yorum satırları #

  CV/DPV:   Potential(V),  Current(A)
  EIS:      Freq(Hz),      Z_real(Ohm),    Z_imag(Ohm)
  pH tablo: pH,            Ep_V,            ip_A
  Kalibr.:  Concentration, ip_run1, ip_run2, ip_run3
=============================================================
"""

import os
import sys

# Modülleri içe aktar
from cv_module         import run_cv_analysis
from eis_module        import run_eis_analysis
from ph_module         import run_ph_analysis
from analytical_module import run_analytical_performance

# ═══════════════════════════════════════════════════════════
# ► BURAYA KENDİ DOSYA YOLLARINI VE PARAMETRELERİNİ GİR ◄
# ═══════════════════════════════════════════════════════════

# ── GENEL PARAMETRELER ────────────────────────────────────
ANALYTE_NAME      = 'Dopamine'       # analiz ettiğin madde
UNIT              = 'μM'              # konsantrasyon birimi
ELECTRODE_AREA    = 0.0707            # GCE standart: 0.0707 cm² (r=1.5mm)
N_ELECTRONS       = 2                 # elektron sayısı
REDOX_PROBE_MM    = 1.0               # [Fe(CN)6] konsantrasyonu, mM
DIFFUSION_COEFF   = 7.6e-6            # [Fe(CN)6] difüzyon katsayısı, cm²/s

# ── CV DOSYALARI ─────────────────────────────────────────
CV_FILES = {
    10:  'data/examples/cv_10mVs.csv',
    25:  'data/examples/cv_25mVs.csv',
    50:  'data/examples/cv_50mVs.csv',
    75:  'data/examples/cv_75mVs.csv',
    100: 'data/examples/cv_100mVs.csv',
    150: 'data/examples/cv_150mVs.csv',
    200: 'data/examples/cv_200mVs.csv',
}

# ── EIS DOSYALARI (modifikasyon adımları sırayla) ────────
EIS_FILES = {
    'Bare GCE':       'data/examples/eis_bare.csv',
    'GCE/Material 1': 'data/examples/eis_step1.csv',
    'GCE/Sensor':     'data/examples/eis_step2.csv',
}

# ── pH TARAMASI ──────────────────────────────────────────
# SEÇENEK A: Her pH için ayrı DPV/CV dosyası
PH_FILES = {
    4.0: 'data/examples/ph_4.csv',
    5.0: 'data/examples/ph_5.csv',
    6.0: 'data/examples/ph_6.csv',
    7.0: 'data/examples/ph_7.csv',
    7.4: 'data/examples/ph_74.csv',
    8.0: 'data/examples/ph_8.csv',
}
# SEÇENEK B: Tablo CSV (pH, Ep_V, ip_A)
# PH_FILES = 'data/ph_table.csv'

# ── KALİBRASYON VERİSİ ──────────────────────────────────
# Dict formatı: {konsantrasyon_μM: [ip_uA tekrar1, tekrar2, tekrar3]}
CALIBRATION_DATA = {
    # ↓ Kendi değerlerinle değiştir
    0.1:  [0.15, 0.16, 0.14],
    0.5:  [0.72, 0.74, 0.71],
    1.0:  [1.43, 1.46, 1.41],
    5.0:  [7.21, 7.18, 7.25],
    10.0: [14.4, 14.5, 14.3],
    50.0: [71.8, 72.1, 71.5],
   100.0: [143.5, 144.0, 143.0],
}
# VEYA CSV dosyası:
# CALIBRATION_DATA = 'data/calibration.csv'

# ── %RSD VERİSİ ──────────────────────────────────────────
RSD_DATA = {
    'Intra-day': {
        'n=1 (10 μM)': [14.4, 14.5, 14.3, 14.6, 14.4]
    },
    'Inter-day': {
        'Day 1': [14.4, 14.5],
        'Day 2': [14.3, 14.6],
        'Day 3': [14.5, 14.4],
    }
}

# ── GERİ KAZANIM VERİSİ ──────────────────────────────────
RECOVERY_DATA = {
    'Matriksler':  ['Blood serum', 'Urine', 'Tap water'],
    'spike_uM':    [10.0, 10.0, 10.0],
    'ip_uA':       [
        [14.3, 14.4, 14.5],
        [14.6, 14.7, 14.5],
        [14.4, 14.3, 14.5],
    ]
}

# ── STANDART İLAVE ───────────────────────────────────────
STANDARD_ADDITION = {
    'additions_uM': [0, 2, 4, 6, 8, 10],
    'ip_uA':        [5.2, 8.0, 10.9, 13.8, 16.7, 19.6],
}

# ═══════════════════════════════════════════════════════════
# ÇALIŞTIR
# ═══════════════════════════════════════════════════════════

def check_files(file_dict):
    """Dosyaların varlığını kontrol et."""
    if isinstance(file_dict, str):
        return os.path.exists(file_dict)
    missing = [p for p in file_dict.values() if not os.path.exists(p)]
    if missing:
        print(f"  ⚠ Eksik dosyalar: {missing}")
        return False
    return True

def main():
    print("\n" + "█"*60)
    print("  ELEKTROKİMYASAL SENSÖR HESAPLAMA ARACI")
    print("  Ankara University / FS3P Lab")
    print("█"*60)

    all_results = {}

    # ── 1. CV ANALİZİ ────────────────────────────────────
    print("\n[1/4] CV ANALİZİ")
    if check_files(CV_FILES):
        cv_results = run_cv_analysis(
            file_dict=CV_FILES,
            concentration_mM=REDOX_PROBE_MM,
            diffusion_coeff=DIFFUSION_COEFF,
            n_electrons=N_ELECTRONS,
            output_dir='results/cv'
        )
        all_results['cv'] = cv_results
    else:
        print("  CV dosyaları bulunamadı — atlanıyor")

    # ── 2. EIS ANALİZİ ───────────────────────────────────
    print("\n[2/4] EIS ANALİZİ")
    if check_files(EIS_FILES):
        eis_results = run_eis_analysis(
            file_dict=EIS_FILES,
            output_dir='results/eis',
            sign_zimag=-1
        )
        all_results['eis'] = eis_results
    else:
        print("  EIS dosyaları bulunamadı — atlanıyor")

    # ── 3. pH TARAMASI ───────────────────────────────────
    print("\n[3/4] pH TARAMASI")
    ph_source = PH_FILES
    if isinstance(ph_source, str):
        exists = os.path.exists(ph_source)
    else:
        exists = check_files(ph_source)
    if exists:
        ph_results = run_ph_analysis(
            source=PH_FILES,
            n_electrons=N_ELECTRONS,
            peak_type='anodic',
            output_dir='results/ph'
        )
        all_results['ph'] = ph_results
    else:
        print("  pH dosyaları bulunamadı — atlanıyor")

    # ── 4. ANALİTİK PERFORMANS ───────────────────────────
    print("\n[4/4] ANALİTİK PERFORMANS")
    analytical_results = run_analytical_performance(
        calibration_source=CALIBRATION_DATA,
        analyte_name=ANALYTE_NAME,
        electrode_area_cm2=ELECTRODE_AREA,
        unit=UNIT,
        rsd_data=RSD_DATA,
        recovery_data=RECOVERY_DATA,
        std_addition_data=STANDARD_ADDITION,
        output_dir='results/analytical'
    )
    all_results['analytical'] = analytical_results

    # ── GENEL ÖZET ───────────────────────────────────────
    print("\n" + "█"*60)
    print("  TÜM ANALİZLER TAMAMLANDI")
    print("█"*60)
    print("\nSonuçlar:")
    for folder in ['results/cv', 'results/eis', 'results/ph', 'results/analytical']:
        if os.path.exists(folder):
            files = os.listdir(folder)
            print(f"  {folder}/  ({len(files)} dosya)")
    print()

    return all_results


if __name__ == '__main__':
    results = main()
