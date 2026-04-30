<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>Universiteti i Prishtinës</p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p>Profesor: Prof. Dr. Kadri Sylejmani</p>
      <p>Asistent: MSc. Labeat Arbneshi</p>
    </td>
 </tr>
</table>

# Optimizimi i Orarit Televiziv — Algoritmi Gjenetik Hibrid

## Përshkrimi i Projektit

Ky projekt implementon një **Algoritëm Gjenetik Hibrid** (Hybrid GA) për
**Problemin e Planifikimit Televiziv për Hapësira Publike** (TV Channel Scheduling
Optimization). Algoritmi quhet **ApexUpgradedScheduler** dhe kombinon:

- **GRASP** (Greedy Randomized Adaptive Search Procedure) për ndërtimin e popullatës fillestare
- **Beam Search** për kërkim konstruktiv në instanca të vogla
- **Algoritëm Gjenetik** me crossover dhe mutacion të përshtatur
- **Large Neighborhood Search (LNS)** si operator mutacioni destroy–rebuild

Qëllimi: **maksimizimi i pikëve totale të shikueshmërisë** duke respektuar
kufizimet kohore, penalitetet e ndërrimit të kanaleve, kohëzgjatjen minimale,
kufizimin e genre-ve të njëpasnjëshme, blloqet prioritare dhe preferencat kohore.

Dokumentimi akademik i plotë i parametrave, operatorëve dhe buxhetit kohor:
**[PARAMETRAT_GA.md](PARAMETRAT_GA.md)** (GA klasik + parametra problem-specifik).

---

## Si funksionon algoritmi (përmbledhje)

`generate_solution()` ndjek një pipeline **anytime** (kohë e kufizuar, jo numër fiks
gjeneratash):

1. **`_seed()` — rreth 55% e kohës totale** (mbetja pas `ga_time_fraction = 0.45`):
   ndërtohet popullata fillestare me **Beam Search** (instanca të vogla; për >100
   kanale beam width për seed = 0 dhe kalon në GRASP) dhe **GRASP** me `alpha` të
   ndryshëm, brenda buxhetit kohor të fazës së ndërtimit.
2. **`_postprocess_elite()`**: mbi individët më të mirë (top-8 / ~25%) aplikohet
   **elite intensification** me destroy–rebuild.
3. **`_ga()` — pjesa tjetër e buxhetit deri në fund**: turnament, crossover (block 88% +
   one-point 12%), mutacion (destroy–rebuild 78% + klasik 22%), elitizëm **20%**
   e popullatës (ose `elite_fraction` i dhënë). Cikli vazhdon deri në skadimin e
   kohës ose **stagnacion** (`stale_lim` gjenerata pa përmirësim — shih
   `PARAMETRAT_GA.md`, §8–9).
4. **`_final_polish()`**: në rezervën e fundit (**~4 s**) polishing me disa raunde
   destroy–rebuild mbi zgjidhjen më të mirë.

Fitness-i është **maksimizim** i `total_score` (shuma e segmentëve: score programi,
bonus preferencash, minus penalitete switch/termination). Shpjegim i plotë:
seksionet A–D në `PARAMETRAT_GA.md`.

---

## Struktura e Projektit

```
├── main.py                              # Hyrja kryesore (vetëm ApexUpgradedScheduler)
├── run_experiments.py                   # Skripti i eksperimenteve (4 konfig x 17 instanca x 10 seeds)
├── scheduler/
│   ├── apex_upgraded_scheduler.py       # Algoritmi kryesor (GA hibrid)
│   ├── apex_scheduler.py               # Klasa prind (GA loop, operatorë, seed)
│   └── beam_search_scheduler.py         # Baza (beam search, kandidatë, skor)
├── models/                              # Modelet e të dhënave
├── parser/                              # Leximi i JSON input
├── serializer/                          # Shkrimi i JSON output
├── utils/                               # Utility
├── data/
│   ├── input/                           # 17 instanca JSON
│   └── output/                          # Zgjidhjet e gjeneruara (JSON)
├── results/                             # CSV me rezultate eksperimentesh
├── PARAMETRAT_GA.md                     # Dokumentacion i detajuar i parametrave
└── README.md                            # Ky dokument
```

---

## Ekzekutimi

```bash
# Ekzekutim i vetëm (shënim: default-et e main.py NUK janë të njëjta me PARAMETRAT_GA;
# për vlera si në dokument / eksperimente, përdorni flag-et eksplicite më poshtë)
python main.py --input data/input/toy.json

# Vlera në përputhje me linjën bazë të dokumentuar (population 60, Pc/Pm/tournament si në PARAMETRAT_GA)
python main.py --input data/input/canada_pw.json \
    --ga-population 60 \
    --ga-crossover-rate 0.55 \
    --ga-mutation-rate 0.50 \
    --ga-tournament 5 \
    --ga-elite 12 \
    --time-limit 300 \
    --seed 42

# Eksperimentet (4 konfigurime fikse; shih seksionin "Eksperimentet")
python run_experiments.py
```

---

## Parametrat e Algoritmit Gjenetik

**Burimi kryesor:** **[PARAMETRAT_GA.md](PARAMETRAT_GA.md)** — tabela përmbledhëse (23
parametra), operatorët (block crossover, destroy–rebuild, tournament), kriteret e
ndalimit (`time_limit_seconds` 300 s, `stale_lim` 50/60, pa target fitness), dhe
parametrat e nxjerrë nga magic numbers (`crossover_rate`, `mutation_rate`,
`tournament_k`, `elite_fraction`, `stale_limit`, `destroy_rebuild_rate`,
`block_crossover_rate`, `polish_rounds`).

**Shpjegim i detajuar:** në **[PARAMETRAT_GA.md](PARAMETRAT_GA.md)** — **seksioni F**
(*Çfarë ndodh nëse ndryshojmë vlerat e parametrave*): për çdo parametër kryesor,
çfarë është, efektet nëse **rritet** ose **ulët** vlera, rreziqet në ekstrem, dhe
lidhja me kohën, diversitetin dhe fitness-in.

### Parametrat kryesorë në CLI (`main.py`)

| Parametri (kodi) | CLI Flag | Shënim |
|------------------|----------|--------|
| `population_size` | `--ga-population` | Në kodin e scheduler-it default tipik **60** (adaptive 15–60 sipas kanaleve); `main.py` ka default tjetër — përdorni flag për përputhje me dokumentin. |
| `crossover_rate` | `--ga-crossover-rate` | Adaptive në kod kur `None`: 0.55 (>100 ch) / 0.65 (≤100 ch). |
| `mutation_rate` | `--ga-mutation-rate` | Adaptive kur `None`: 0.50 / 0.40. |
| `tournament_k` | `--ga-tournament` | Adaptive kur `None`: 5 / 3. |
| `elite_fraction` | `--ga-elite` | Jepet si **numër** elitësh; në kod llogaritet `elite_fraction = ga_elite / population`. Për 20% me pop 60: `--ga-elite 12`. |
| `time_limit` | `--time-limit` | Max 300 s në këtë build. |
| `seed` | `--seed` | Reproduktibilitet (default në dokument: 42 ku përdoret). |
| `lookahead_limit` | `--lookahead` | Default 6 në dokument dhe eksperimente. |
| `density_percentile` | `--density-percentile` | Default 25. |

`destroy_rebuild_rate`, `block_crossover_rate` dhe `ga_time_fraction` **nuk** janë
ekspozuar në `main.py`; në eksperimente vendosen në `run_experiments.py` te
`ApexUpgradedScheduler(...)`.

### Zgjidhja fillestare (cf. PARAMETRAT_GA, §19 dhe diagrami D)

- **Ndërtimi** përdor Beam Search (me **beam width** adaptive sipas `n_channels`: deri
  në 200 për instanca shumë të vogla) dhe **GRASP** (`alpha` 0–0.50).
- **Local search** (`_local_search`) **është hequr** nga faza e seed-imit: popullata
  fillestare është e udhëhequr nga heuristikat, pa polish greedy shtesë, që GA-ja
  të bëjë optimizimin evolucionar.
- Pjesa kohore e mëparshme e pipeline-it (~**55%** konstruksion, **45%** GA) përputhet
  me `ga_time_fraction = 0.45`.

---

## Eksperimentet

### 4 Konfigurimet e Testuara

Të gjitha përdorin të njëjtat vlera të fiksuara si eksperiment: `TIME_LIMIT = 300`,
`ga_time_fraction = 0.45`, `lookahead_limit = 6`, `density_percentile = 25`, dhe
`block_crossover_rate = 0.88` kudo (operatori i njëjtë i crossover në bllok).

| Parametri | Default | Explorues | Intensiv | Balancuar |
|-----------|---------|-----------|----------|-----------|
| `population_size` | 60 | 80 | 40 | 60 |
| `crossover_rate` | 0.55 | 0.40 | 0.80 | 0.65 |
| `mutation_rate` | 0.50 | 0.70 | 0.25 | 0.45 |
| `tournament_k` | 5 | 3 | 7 | 5 |
| `elite_fraction` | 0.20 | 0.10 | 0.30 | 0.20 |
| `destroy_rebuild_rate` | 0.78 | 0.78 | 0.78 | 0.90 |
| `block_crossover_rate` | 0.88 | 0.88 | 0.88 | 0.88 |

**Ndryshimet krahasuar me Default** (çfarë ndryshon vetëm një kolonë):

| Konfigurimi | Nga Default ndryshon |
|-------------|----------------------|
| **Explorues** | +20 individë; Pc −0.15; Pm +0.20; *k* −2; elitë −10 p.p. → më shumë diversitet, më pak kombinim agresiv prindërish, më shumë mutacion. |
| **Intensiv** | −20 individë; Pc +0.25; Pm −0.25; *k* +2; elitë +10 p.p. → popullatë e vogël e presionuar, më shumë crossover dhe më pak mutacion, seleksion më i ashpër. |
| **Balancuar** | Pm −0.05; Pc +0.10 (si instancë e vogël në adaptive); `destroy_rebuild_rate` +0.12 → njësoj madhësi popullate dhe elitë si Default, por crossover më “i hapur” dhe mutacion pak më i ulët; kur ndodh mutacioni, më shpesh **LNS destroy–rebuild** (0.90) se sa në Default (0.78). |

Interpretrimi i shkurtër:

- **Default** — linja bazë: konstante eksplicite (~vlerat adaptive të dokumentuara për instanca të mëdha: pop 60, Pc 0.55, Pm 0.50, *k* 5, elitë 20%); përputhet me `CONFIGS["Default"]` në `run_experiments.py`.
- **Explorues** — eksplorim: popullatë më e madhe, Pm i lartë, Pc i ulët, turnament i butë (*k*=3), elitë e vogël.
- **Intensiv** — eksploatim: popullatë e vogël, Pm i ulët, Pc i lartë, turnament i fortë (*k*=7), elitë e madhe.
- **Balancuar** — si Default për shumicën e GA-së; dallon kryesisht nga **rritja e shpeshtësisë së destroy–rebuild** kur aplikohet mutacioni (`destroy_rebuild_rate` 0.90).

#### Roli i parametrave (përmbledhje)

| Parametri | Roli në GA / hibrid |
|-----------|---------------------|
| `population_size` | Numër individësh për gjeneratë; më i madh → më shumë diversitet por më shumë kosto për gjeneratë. |
| `crossover_rate` | Sa shpesh gjenerohen fëmijë me crossover në vend të kopjes së drejtpërdrejtë; më i lartë → më shumë kombinim i prindërve. |
| `mutation_rate` | Sa shpesh aplikohet mutacioni pas crossover-it; më i lartë → më shumë eksplorim, më i ulët → më shumë stabilitet rreth zgjidhjeve të mira. |
| `tournament_k` | Në turnament, sa individë nxirren për të zgjedhur prindin; *k* më i madh → presion selektues më i fortë (më shumë “fitnes”). |
| `elite_fraction` | Pjesa e popullatës që kalon pa ndryshim në gjeneratën tjetër; më e lartë → më shumë ruajtje e elitës. |
| `destroy_rebuild_rate` | (Vetëm në `ApexUpgradedScheduler`.) Në mutacion, probabiliteti të përdoret *destroy–rebuild* në një dritare kohe (LNS) në vend të mutacionit të `ApexScheduler`. |
| `block_crossover_rate` | Në crossover, probabiliteti të përdoret crossover në bllokë programesh në vend të crossover-it standard të prindit. |

Në të katër konfigurimet `block_crossover_rate` është i njëjtë (0.88), pra eksperimentet nuk krahasojnë ndryshimin e këtij operatori, vetëm GA + `destroy_rebuild` sipas rreshtit të mësipërm.

### Protokolli

- **17 instanca** (toy, canada_pw, usa_tv, uk_tv, croatia_tv, germany_tv, kosovo_tv, china_pw, youtube_premium, youtube_gold, us_iptv, uk_iptv, spain_iptv, france_iptv, australia_iptv, singapore_pw, netherlands_tv)
- **10 ekzekutime** për secilën instancë (seed 1–10)
- **300 sekonda** (5 minuta) buxhet kohor për ekzekutim
- **Total: 680 ekzekutime** (4 × 17 × 10)

### Rezultatet

Rezultatet e plota ndodhen në `results/all_results.csv` dhe `results/summary.csv`.

Per cdo ekzekutim ruhet edhe zgjidhja JSON në `data/output/`.

<!-- RESULTS_TABLE_START -->
*Tabela e rezultateve do plotësohet pas përfundimit të eksperimenteve.*
<!-- RESULTS_TABLE_END -->

---

## Referencat

1. **Eiben, A.E. & Smith, J.E.** (2015). *Introduction to Evolutionary Computing*. Springer.
2. **De Jong, K.A.** (1975). *Analysis of the behavior of a class of genetic adaptive systems*. PhD Thesis, University of Michigan.
3. **Goldberg, D.E. & Deb, K.** (1991). "A comparative analysis of selection schemes used in genetic algorithms." *Foundations of Genetic Algorithms*.
4. **Radcliffe, N.J.** (1994). "The algebra of genetic algorithms." *Annals of Mathematics and AI*.
5. **Shaw, P.** (1998). "Using constraint programming and local search methods to solve vehicle routing problems." *CP-98*.
6. **Ropke, S. & Pisinger, D.** (2006). "An Adaptive Large Neighborhood Search Heuristic." *Transportation Science*.
7. **Feo, T.A. & Resende, M.G.C.** (1995). "Greedy randomized adaptive search procedures." *Journal of Global Optimization*.
