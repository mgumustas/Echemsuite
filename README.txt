════════════════════════════════════════════════════════
ELEKTROKİMYASAL SENSÖR HESAPLAMA ARACI
Ankara University / FS3P Lab
════════════════════════════════════════════════════════

KURULUM (bir kez):
  pip install numpy pandas matplotlib scipy

ÇALIŞTIRMA:
  cd echem_calc/
  python run_analysis.py

─────────────────────────────────────────────────────
DOSYA YAPISI
─────────────────────────────────────────────────────
  run_analysis.py          ← Buradan çalıştır, parametreleri buraya gir
  cv_module.py             ← CV hesapları (ECSA, mekanizma, k°)
  eis_module.py            ← EIS fitting (Randles-CPE)
  ph_module.py             ← pH taraması (Nernst eğimi, m/n)
  analytical_module.py     ← LOD/LOQ, kalibrasyon, recovery
  data/examples/           ← Test için örnek CSV dosyaları
  results/                 ← Çıktı grafikleri ve tablolar (otomatik)

─────────────────────────────────────────────────────
KULLANIM — KISA VERSİYON
─────────────────────────────────────────────────────

1. run_analysis.py dosyasını aç
2. "BURAYA KENDİ DOSYA YOLLARINI GİR" bölümünü bul
3. Kendi CSV dosyalarının yollarını yaz
4. n_electrons, ELECTRODE_AREA gibi parametreleri ayarla
5. python run_analysis.py

Sadece bir modül çalıştırmak için:
  python cv_module.py
  python eis_module.py
  python ph_module.py
  python analytical_module.py

─────────────────────────────────────────────────────
GİRİŞ CSV FORMATLARI
─────────────────────────────────────────────────────

CV / DPV dosyası:
  Potential(V),Current(A)
  0.100,1.23e-7
  0.105,1.45e-7
  ...

EIS dosyası:
  Freq(Hz),Z_real(Ohm),Z_imag(Ohm)
  100000,10.5,-0.8
  50000,10.6,-1.2
  ...
  NOT: Z_imag çoğu potansiyostatta negatif verilir (sign_zimag=-1)

pH tablo dosyası (hazır tablo için):
  pH,Ep_V,ip_A
  4.0,0.330,3.2e-6
  5.0,0.270,5.8e-6
  ...

Kalibrasyon dosyası:
  Concentration(uM),Run1,Run2,Run3
  0.1,0.15,0.16,0.14
  0.5,0.72,0.74,0.71
  ...

─────────────────────────────────────────────────────
HESAPLAMA ÇIKTILARI
─────────────────────────────────────────────────────

CV:
  - ECSA (cm²) — Randles-Ševčík
  - b eğimi — difüzyon/adsorpsiyon kontrolü
  - k° (cm/s) — Nicholson metodu
  - Pik tablosu: Epa, ipa, Epc, ipc, ΔEp

EIS:
  - Rs, Rct, CPE-Q, CPE-n, σ (Warburg)
  - C_eff (efektif kapasitans)
  - χ² (fit kalitesi)
  - Pürüzlülük faktörü (RF)

pH:
  - Nernst eğimi (mV/pH)
  - m/n proton-elektron oranı
  - Optimal pH

Analitik:
  - Kalibrasyon denklemi, r, R²
  - LOD, LOQ (3σ/m, 10σ/m)
  - Normalize hassasiyet (μA/μM/cm²)
  - %RSD (intra-day, inter-day)
  - Geri kazanım (Recovery%)
  - Standart ilave analizi

─────────────────────────────────────────────────────
YAYINDA KULLANIM
─────────────────────────────────────────────────────

Hesaplanan tüm değerleri makalenin ilgili bölümlerine aktar:

  Karakterizasyon:
    EIS → Tablo 1 (Rs, Rct, Q, n, χ²)
    CV  → Şekil açıklamalarına ECSA, k°, b eğimi

  Analitik Performans:
    → Tablo 2: Lineer aralık, LOD, LOQ, hassasiyet, %RSD
    → Tablo 3: Recovery tablosu

  "LOD was calculated as 3σ/m, where σ is the standard
  deviation of the blank signal and m is the slope of
  the calibration curve."

─────────────────────────────────────────────────────
POTANSİYOSTAT UYUMLULUĞU
─────────────────────────────────────────────────────

Test edilmiş/uyumlu CSV formatları:
  - Gamry (CSV dışa aktarma)
  - Metrohm Autolab
  - CH Instruments
  - PalmSens

Eğer veri yüklenemiyorsa:
  1. CSV'yi Excel'de aç
  2. Potential(V), Current(A) sütunlarını kontrol et
  3. Ondalık ayırıcının nokta (.) olduğunu doğrula
  4. Başlık satırını # ile başlatarak yorum satırına al

════════════════════════════════════════════════════════
