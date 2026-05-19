# Žmogaus judesių atpažinimo ir grįžtamojo ryšio sistema

Krepšinio metimo analizės sistema, pagrįsta žmogaus kūno laikysenos įvertinimo algoritmu. Sistema realiuoju laiku aptinka kūno taškus, analizuoja metimo techniką ir teikia grįžtamąjį ryšį.

---

## Reikalavimai

- Python 3.10+
- Kamera (numatytoji: kamera, `--source 4`; nešiojamam kompiuteriui – `--source 0`)

---

## Įdiegimas

```bash
git clone https://github.com/Lauraven/HUMAN-MOTION-RECOGNITION-AND-FEEDBACK-SYSTEM-BASED-ON-HUMAN-POSE-ESTIMATION-ALGORITHM.git
cd HUMAN-MOTION-RECOGNITION-AND-FEEDBACK-SYSTEM-BASED-ON-HUMAN-POSE-ESTIMATION-ALGORITHM

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Priklausomybės

| Biblioteka | Paskirtis |
|---|---|
| `opencv-python` | Vaizdo apdorojimas |
| `mediapipe` | Kūno taškų aptikimas |
| `kivy` | Vartotojo sąsaja |
| `numpy` | Skaitiniai skaičiavimai |
| `pandas`, `openpyxl` | Duomenų įrašymas |
| `matplotlib`, `scipy` | Grafikai |

---

## Paleidimas

```bash
# Orbbec kamera
python main.py

python video_demo.py "path to the video\video.mp4" --out "path to save results\video_rez.mp4"
```

---

## 🎮 Valdymas

| Klavišas | Veiksmas |
|---|---|
| `Q` | Išeiti |
| `R` | Atstatyti skaitliukus |
| `F` | Įjungti / išjungti veidrodį |
| `P` | Rodyti grafikus |

---

## Projekto struktūra

```
├── main.py              # Pagrindinis paleidimo failas
├── process_frame.py     # Kadro apdorojimas ir fazių aptikimas
├── ui.py                # Kivy vartotojo sąsaja
├── data_logger.py       # Duomenų įrašymas
├── plotter.py           # Grafikų generavimas
├── utils.py             # Pagalbinės funkcijos
├── ball.py              # Kamuolio sekimas
├── good_shot_data.py    # Etalono duomenys
├── pose_landmarker_full.task  # MediaPipe modelis
├── requirements.txt     # Priklausomybės
└── video_demo/          # Demonstraciniai vaizdo įrašai
```
