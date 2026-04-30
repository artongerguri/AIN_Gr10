# Analiza e Plotë e Parametrave të Algoritmit Gjenetik
## ApexUpgradedScheduler — Niveli Master

---

## Hyrje

`ApexUpgradedScheduler` implementon një **algoritëm gjenetik hibrid** (Hybrid GA) për
problemin e planifikimit optimal të orarit televiziv. Algoritmi kombinon:

- **Ndërtim greedy heuristik** (GRASP — Greedy Randomized Adaptive Search Procedure)
- **Beam Search** për kërkim konstruktiv
- **Algoritëm Gjenetik** (GA) për optimizim evolucionar
- **Kërkim lokal** përmes destroy–rebuild (Large Neighborhood Search)

Më poshtë identifikohen dhe shpjegohen **të gjitha parametrat** e përdorura, duke i
ndarë në dy kategori: (A) parametra direkt të GA-së dhe (B) parametra shtesë
problem-specifik që ndikojnë në sjelljen e algoritmit.

---

## A. PARAMETRAT E DREJTPËRDREJTË TË ALGORITMIT GJENETIK

---

### 1. Madhësia e Popullatës (Population Size) — `population_size`

| Aspekti | Vlera |
|---------|-------|
| **Default** | **60** |
| **Vendndodhja** | `ApexUpgradedScheduler.__init__()`, rreshti 35 |
| **Përshtatje adaptive** | Po — zvogëlohet sipas numrit të kanaleve |

**Si funksionon në kod:**

```
if n_channels > 5000:  pop_size = min(60, 15)   →  15
if n_channels > 2000:  pop_size = min(60, 25)   →  25
if n_channels > 500:   pop_size = min(60, 40)   →  40
if n_channels ≤ 500:   pop_size = 60            →  60
```

**Arsyeja akademike:** Popullatë më e madhe rrit diversitetin gjenetik dhe zvogëlon
rrezikun e konvergjencës së parakohshme (premature convergence), por rrit koston
llogaritëse për gjeneratë. Në instanca të mëdha (>5000 kanale), çdo individ kërkon
shumë kohë për t'u ndërtuar, kështu popullatë e vogël (15) parandalon tejkalimin e
buxhetit kohor. Kjo përshtatje adaptive i referohet konceptit të **parameter tuning
at runtime** (Eiben et al., 2007).

---

### 2. Probabiliteti i Crossover-it (Crossover Rate) — `cx_rate`

| Aspekti | Vlera |
|---------|-------|
| **Instanca e madhe** (>100 kanale) | **0.55** |
| **Instanca e vogël** (≤100 kanale) | **0.65** |
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 662–663 |

**Si funksionon në kod:**

```python
if self.rng.random() < cx_rate:
    child = self._crossover(p1, p2)
else:
    child = max(p1, p2, key=lambda s: s.total_score)  # asexual reproduction
```

Kur crossover nuk ndodh, fëmija trashëgon prindin me fitness më të lartë —
ky mekanizëm quhet **asexual reproduction** ose **cloning**.

**Arsyeja akademike:** Vlera 0.55–0.65 bie brenda diapazonit standard
(0.6–0.9, De Jong 1975), por është pak më e ulët sepse mutacioni
destroy–rebuild në këtë algoritëm është shumë agresiv dhe luan rolin e eksplorimeve.
Një Pc më e ulët i jep më shumë hapësirë mutacionit.

---

### 3. Probabiliteti i Mutacionit (Mutation Rate) — `mut_rate`

| Aspekti | Vlera |
|---------|-------|
| **Instanca e madhe** | **0.50** (50%) |
| **Instanca e vogël** | **0.40** (40%) |
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 662–663 |

**Si funksionon në kod:**

```python
if self.rng.random() < mut_rate:
    child = self._mutate(child)
```

Mutacioni aplikohet **pas** crossover-it (ose cloning-ut). Kjo do të thotë se
një fëmijë mund të ketë kaluar edhe crossover edhe mutacion në të njëjtën gjeneratë.

**Arsyeja akademike:** Vlera 40–50% është **shumë më e lartë** se norma klasike
(0.001–0.1 për encoding binar). Kjo arsyetohet me natyrën e mutacionit:

- Nuk bëhet flip-bit por **destroy–rebuild** — një formë e Large Neighborhood
  Search (LNS, Shaw 1998) e integruar si operator mutacioni.
- Ky lloj mutacioni rindërton vetëm një dritare (window) të vogël, pra është
  një lëvizje e kontrolluar, jo kaotike.
- Vlera e lartë e Pm mundëson diversitet të vazhdueshëm pa rrezikuar
  destabilizim, sepse riparimi (`_repair`) siguron fizibilitetin.

---

### 4. Funksioni i Fitness-it (Fitness Function) — `total_score`

| Aspekti | Detaji |
|---------|--------|
| **Tipi** | Maksimizim |
| **Vlera** | Shumë e pikëve të secilit segment (`fitness`) |
| **Vendndodhja** | `_calc_score()` në `BeamSearchScheduler`, rreshti 158–204 |

**Formula e plotë:**

```
fitness(segment) = score_programi
                 + bonus_preferencës          (nëse genre përputhet me dritaren kohore)
                 − switch_penalty             (nëse ndryshon kanali)
                 − termination_penalty         (nëse fillon vonë — late start)
                 − termination_penalty         (nëse ndërpret herët — early stop)
```

```
total_score(Solution) = Σ fitness(segment_i)  për çdo segment i
```

**Arsyeja akademike:** Fitness-i agregon në një skalar të vetëm disa objektiva
(maksimizim i shikueshmërisë, minimizim i penaliteteve). Kjo përqasje quhet
**scalarization** e problemeve shumë-objektive (Deb, 2001). Penalitetet e
integruara (switch, termination) shërbejnë si **penalty-based constraint handling**
— shkeljet e rregullave nuk e bëjnë zgjidhjen jovalide, por e ulin fitness-in.

---

### 5. Metoda e Seleksionimit (Selection Method) — Tournament Selection

| Aspekti | Vlera |
|---------|-------|
| **Metoda** | **Tournament Selection** |
| **k (madhësia e turnamentit)** | **5** (e madhe) ose **3** (e vogël) |
| **Vendndodhja** | `ApexScheduler._tournament()`, rreshti 551–554 |

**Si funksionon në kod:**

```python
def _tournament(self, pop, k=5):
    return max(
        self.rng.sample(pop, min(k, len(pop))),
        key=lambda s: s.total_score
    )
```

Thirret dy herë për çdo fëmijë (zgjedh `p1` dhe `p2`), pra kemi
**presion seleksionimi simetrik**.

**Arsyeja akademike:**

- **k=5** → presion i lartë seleksionimi → favorizon eksploatimin (exploitation).
  Kjo justifikohet sepse mutacioni i fortë (40–50%) siguron eksplorim (exploration).
- Tournament Selection ka **kompleksitet O(k)** dhe nuk kërkon renditje
  të popullatës, ndryshe nga Rank Selection ose Roulette Wheel.
- Nuk kërkon normalizim si Roulette Wheel (Goldberg & Deb, 1991), që mund
  të shkaktojë scaling bias me vlera negative të fitness-it.

---

### 6. Metoda e Crossover-it — Block Crossover (Hibrid)

| Aspekti | Vlera |
|---------|-------|
| **Vendndodhja** | `ApexUpgradedScheduler._crossover()` + `_block_crossover()` |
| **Probabiliteti i block crossover** | **88%** (0.88) |
| **Probabiliteti i one-point fallback** | **12%** (0.12) |

**Dy variante:**

**(a) Block Crossover (88% e rasteve)** — `_block_crossover()`:
- Identifikon **blloqet zinxhir** (programet e njëpasnjëshme në të njëjtin kanal)
  përmes `_identify_blocks()`
- Zgjedh pikën e prerjes ose në kufi blloku (62%) ose në kohë natyrale
  (kufi i priority block-ut, 38%)
- Prefix nga prindi 1, suffix i filtruar nga prindi 2 (pa duplikime, pa
  mbivendosje kohore)
- Riparimi (`_repair`) siguron fizibilitetin

**(b) One-Point Crossover (12% e rasteve)** — `ApexScheduler._crossover()`:
- Zgjedh indeks random `cut` në schedulin e prindit 1
- Prefix: `s1[:cut]`, suffix: segmentet e `s2` pas kohës cut_time
- Pastaj `_repair(prefix + suffix)`

**Arsyeja akademike:**

Block Crossover është **structure-preserving crossover** — ruan zinxhirët
(continuation chains) të cilët kanë vlerë të lartë sepse eliminojnë penalitetet
switch/termination. Kjo i referohet konceptit të **respectful recombination**
(Radcliffe, 1994): operatori nuk shkatërron tipare (building blocks) që janë
të pranishme te të dy prindërit. Varianti hibrid (12% one-point) siguron
diversitet structural.

---

### 7. Metoda e Mutacionit — Destroy–Rebuild (Large Neighborhood Search)

| Aspekti | Vlera |
|---------|-------|
| **Vendndodhja** | `ApexUpgradedScheduler._mutate()` + `_destroy_rebuild_mutation()` |
| **Probabiliteti destroy–rebuild** | **78%** |
| **Probabiliteti mutacion klasik** | **22%** (bie te `ApexScheduler._mutate()`) |

**Katër mënyra zgjedhjeje të dritares për shkatërrim:**

| Mënyra | Probabiliteti | Përshkrimi |
|--------|---------------|------------|
| Weak Region | 28% | Zona me fitness mesatar më të ulët |
| Switch-Heavy Region | 24% | Zona me më shumë ndërrime kanalesh |
| Preference-Adjacent | 20% | Pranë dritareve të preferencave |
| Random | 28% | Pozicion i rastit |

Pas heqjes së segmenteve, rindërtimi bëhet me **Beam Fill Gap** — beam search
i kufizuar brenda dritares kohore.

**Arsyeja akademike:** Ky operator është adaptim i **Adaptive Large Neighborhood
Search (ALNS)** (Ropke & Pisinger, 2006). Përdorimi i disa heuristikave
shkatërrimi (weak, switch, pref, random) me probabilitete fikse i referohet
**mekanizmave diversifikimi** — secila heuristikë synon optimizim të një
aspekti tjetër të zgjidhjes. Kjo siguron që algoritmi nuk ngec në optima lokale
të një lloji të vetëm.

---

### 8. Numri i Gjeneratave — Dinamik (jo fiks)

| Aspekti | Vlera |
|---------|-------|
| **Kufizim** | Buxhet kohor + stale limit |
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 666 |

**Si funksionon:**

```python
while _time.time() < deadline and stale < stale_lim:
    gen += 1
    ...
```

Nuk ka numër fiks gjeneratash. Cikli vazhdon derisa:
- Të kalojë koha (`deadline`), ose
- Të arrijë kufirin e stagnacionit (`stale_lim`)

**Arsyeja akademike:** Qasja **anytime algorithm** — algoritmi kthen
zgjidhjen më të mirë pavarësisht se kur ndalet. Kjo është superiore ndaj
numrit fiks sepse adaptohet me madhësinë e instancës: instanca të vogla bëjnë
shumë gjenerata, instancat e mëdha pak por me kërkim më të thellë për gjeneratë.

---

### 9. Kriteret e Ndalimit (Stopping Criteria) — Kombinim i trefishtë

| Kriteri | Parametri | Vlera |
|---------|-----------|-------|
| **Buxhet kohor total** | `time_limit_seconds` | **300 sekonda** (5 min) |
| **Buxhet GA** | `ga_time_fraction` | **45%** e kohës totale |
| **Stagnacion** | `stale_lim` | **50** (e madhe) ose **60** (e vogël) gjenerata |

**Si funksionon:**

```
time_limit = 300s
constr_budget ≈ 300 × (1 − 0.45) − 10 = max(5, ~155s − 10s)  →  ndërtimi
ga_budget ≈ koha_e_mbetur − 1.5s                                →  GA
_FINAL_RESERVE_S = 4s                                           →  polish final
```

Nuk përdoret ndalim **"arrihet fitness i caktuar"** — vetëm kohë dhe stagnacion.

**Arsyeja akademike:** Buxheti kohor fiks është standard në **competitive
scheduling** (p.sh. gara algoritmike) ku koha e ekzekutimit është e kufizuar.
Stagnacioni (`stale_lim`) implementon **convergence detection** — nëse
popullatë nuk përmirësohet për N gjenerata, ka konvergjuar.

---

### 10. Elitizmi (Elitism)

| Aspekti | Vlera |
|---------|-------|
| **Numri i elitëve** | `max(2, len(pop) // 5)` → **20% e popullatës** |
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 656 |

**Si funksionon në kod:**

```python
elite_n = max(2, len(pop) // 5)
...
nxt = list(pop[:elite_n])  # kopjo elitët direkt
while len(nxt) < len(pop):
    # krijo fëmijë të rinj...
```

**Shtesë në ApexUpgradedScheduler — `_postprocess_elite()`:**

Pas seed-imit, **top-8 individët** (ose top-25%) i nënshtrohen një faze
shtesë optimizimi destroy–rebuild. Kjo quhet **elite intensification**.

**Arsyeja akademike:** Elitizmi garanton **monotonicitet** — fitness-i më i mirë
nuk bie kurrë ndërmjet gjeneratave (De Jong, 1975). Përqindja 20% është
më agresive se norma (zakonisht 1–2 individë), por justifikohet nga
mutacioni i fortë që siguron diversitet.

---

## B. PARAMETRA SHTESË PROBLEM-SPECIFIK

Këta nuk janë parametra klasik të GA-së, por ndikojnë vendimtarisht në
cilësinë e operatorëve dhe zgjidhjes.

---

### 11. Fraksioni Kohor i GA-së — `ga_time_fraction`

| Vlera | 0.45 (45%) |
|-------|------------|
| **Vendndodhja** | `__init__()`, rreshti 36 |

Ndan buxhetin kohor: **55% ndërtim heuristik** + **45% optimizim evolucionar**.

---

### 12. Parametri GRASP Alpha — `alpha`

| Vlera | 0.0 – 0.50 (shumë vlera) |
|-------|---------------------------|
| **Vendndodhja** | `ApexScheduler._construct()`, rreshti 370–376 |

```python
thr = bv − alpha × (bv − wv)
rcl = [r for r in ranked if r[0] >= thr]
pick = rng.choice(rcl)
```

Kontrollon **randomizimin e listës së kandidatëve të kufizuar (RCL)**:
- `alpha = 0` → greedy i pastër (zgjedh gjithmonë më të mirën)
- `alpha = 0.50` → pranon edhe kandidatë me 50% të dallimit

---

### 13. Seed-i Random — `seed`

| Vlera | Default: **42** |
|-------|-----------------|
| **Vendndodhja** | `ApexScheduler.__init__()`, rreshti 50 |

Mundëson **reproduktibilitet** — e njëjta seed jep të njëjtën zgjidhje.

---

### 14. Lookahead Limit — `lookahead_limit`

| Vlera | **6** |
|-------|-------|
| **Vendndodhja** | `BeamSearchScheduler.__init__()`, rreshti 17 |

Sa programe në të ardhmen (sipas kohës) merren parasysh kur kërkohen
kandidatët. Vlera 6 do të thotë: shiko deri 6 pika kohore përpara.

---

### 15. Density Percentile — `density_percentile`

| Vlera | **25** (top 25%) |
|-------|------------------|
| **Vendndodhja** | `BeamSearchScheduler._preprocess()`, rreshti 121 |

Përcakton `avg_score_per_min` — densiteti mesatar i pikëve bazuar në **top-25%**
e programeve. Përdoret si heuristikë **vlerësimi i ardhshëm (future value
estimation)** në beam search dhe ranking.

---

### 16. Madhësia e Turnamentit — `k`

| Vlera | **5** (instanca e madhe) ose **3** (e vogël) |
|-------|----------------------------------------------|
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 662–663 si `tk` |

Ndikon direkt në **presionin e seleksionimit**: k më i madh → më shumë
eksploatim; k më i vogël → më shumë eksplorim.

---

### 17. Stale Limit — `stale_lim`

| Vlera | **50** (instanca e madhe) ose **60** (e vogël) |
|-------|------------------------------------------------|
| **Vendndodhja** | `ApexScheduler._ga()`, rreshti 662–663 |

Numri i gjeneratave pa përmirësim para se të ndalet GA. Kriter
**konvergjence** (convergence criterion).

---

### 18. Parametrat Adaptiv të Beam-it — `_adaptive_window_params()`

| Parametri | Instanca e vogël (≤100ch) | Instanca e mesme | Instanca e madhe (>5000ch) |
|-----------|---------------------------|-------------------|---------------------------|
| `max_segs` | 6 | 4–5 | 2 |
| `beam_w` | 6 | 4–5 | 2 |
| `max_expansions` | 320 | 120–200 | 40 |
| `cand_cap` | 14 | 14 | 10 |

Kontrollon **gjerësinë dhe thellësinë e beam-it** gjatë fazës
destroy–rebuild. Zvogëlohen kur koha e mbetur bie nën 25 ose 60 sekonda.

---

### 19. Beam Width (për seed-in fillestar) — adaptive

| n_channels | beam_width |
|------------|------------|
| ≤ 20 | 200 |
| ≤ 50 | 150 |
| ≤ 100 | 60 |
| > 100 | 0 (skip) |

Instancat e vogla fillojnë me beam search të pastër; instancat e mëdha
kalojnë direkt në ndërtim GRASP.

**Shënim:** Local search (`_local_search`) është **hequr** nga faza e seed-imit.
Zgjidhja fillestare ndërtohet vetëm me beam search + GRASP heuristik (guided),
pa fazë shtesë lokale greedy. Kjo siguron që popullatë fillestare është e
cilësisë së mirë (guided) por jo e optimizuar lokalisht — GA-ja merr përsipër
përmirësimin evolucionar.

---

### 20. Final Reserve — `_FINAL_RESERVE_S`

| Vlera | **4.0 sekonda** |
|-------|-----------------|
| **Vendndodhja** | `ApexUpgradedScheduler`, rreshti 29 |

Kohë e rezervuar në fund për **polishing** — destroy–rebuild e fundit
mbi zgjidhjen më të mirë.

---

### 21. Penalitetet e Problemit (nga instanca)

| Parametri | Përshkrimi |
|-----------|------------|
| `switch_penalty` | Penalitet kur ndryshon kanali |
| `termination_penalty` | Penalitet për late start ose early stop |
| `max_consecutive_genre` | Nr. maks. segmentesh radhazi me të njëjtin genre |
| `min_duration` | Kohëzgjatja minimale e një segmenti |

Këta vijnë nga instanca e problemit (`InstanceData`), jo nga GA, por
ndikojnë **direkt** në fitness-in dhe në strategjinë e ranking-ut.

---

### 22. Raporti i Penalitetit — `_pen_ratio` (derivuar)

```python
_pen = switch_penalty + 2 × termination_penalty
_pen_ratio = _pen / (avg_score_per_min × min_duration)
```

| pen_ratio | _fw | _fb | Interpretuimi |
|-----------|-----|-----|---------------|
| > 0.6 | 0.55 | 2.5 | Penalitete të larta → favorizon qëndrimin në kanal |
| 0.3–0.6 | 0.70 | 1.5 | Mesatare |
| < 0.3 | 0.85 | 0.8 | Penalitete të ulëta → më fleksibël me ndryshimet |

`_fw` (future weight) dhe `_fb` (full-program bonus) rregullojnë
agresivitetin e ranking-ut.

---

## NDRYSHIMET NË KOD — Parametrat e shtuar

Parametrat e mëposhtëm ishin **hardcoduar si magic numbers** brenda funksioneve.
Tani janë nxjerrë si parametra konfigurueshëm në `__init__()`:

### Në `ApexScheduler.__init__()`:

| Parametri i ri | Tipi | Default | Ku përdorej si magic number |
|----------------|------|---------|---------------------------|
| `crossover_rate` | `Optional[float]` | `None` → 0.55/0.65 adaptive | `_ga()` rreshti 662 |
| `mutation_rate` | `Optional[float]` | `None` → 0.50/0.40 adaptive | `_ga()` rreshti 662 |
| `tournament_k` | `Optional[int]` | `None` → 5/3 adaptive | `_ga()` rreshti 662 |
| `elite_fraction` | `float` | `0.20` (20%) | `_ga()` si `len(pop)//5` |
| `stale_limit` | `Optional[int]` | `None` → 50/60 adaptive | `_ga()` rreshti 662 |

Kur vlera është `None`, përdoret sjellja adaptive origjinale (vlera ndryshon
sipas madhësisë së instancës). Kur jepet vlerë eksplicite, ajo mbizotëron.

### Në `ApexUpgradedScheduler.__init__()`:

| Parametri i ri | Tipi | Default | Ku përdorej si magic number |
|----------------|------|---------|---------------------------|
| `destroy_rebuild_rate` | `float` | `0.78` | `_mutate()` rreshti 390 |
| `block_crossover_rate` | `float` | `0.88` | `_crossover()` rreshti 446 |
| `polish_rounds` | `Optional[int]` | `None` → 2/3 adaptive | `_final_polish()` rreshti 481 |

---

## C. TABELA PËRMBLEDHËSE

| # | Parametri | Vlera | Tip | Konfigurueshëm? | Kategori |
|---|-----------|-------|-----|-----------------|----------|
| 1 | Population Size | 60 (adaptive: 15–60) | `population_size` | PO | GA klasik |
| 2 | Crossover Rate (Pc) | 0.55 / 0.65 | `crossover_rate` | PO (tani) | GA klasik |
| 3 | Mutation Rate (Pm) | 0.50 / 0.40 | `mutation_rate` | PO (tani) | GA klasik |
| 4 | Fitness Function | Σ(score − penalties) | Derivuar | JO (logjikë) | GA klasik |
| 5 | Selection Method | Tournament (k=3/5) | `tournament_k` | PO (tani) | GA klasik |
| 6 | Crossover Method | Block (88%) + One-Point | `block_crossover_rate` | PO (tani) | GA klasik |
| 7 | Mutation Method | Destroy–Rebuild (78%) | `destroy_rebuild_rate` | PO (tani) | GA klasik |
| 8 | Num. Gjeneratave | Dinamik (buxhet kohor) | Anytime | JO | GA klasik |
| 9 | Stopping Criteria | Kohë + Stagnacion | `stale_limit` | PO (tani) | GA klasik |
| 10 | Elitism | 20% e popullatës | `elite_fraction` | PO (tani) | GA klasik |
| 11 | GA Time Fraction | 0.45 | `ga_time_fraction` | PO | Problem-specifik |
| 12 | GRASP Alpha | 0.0–0.50 | Shumë-vlerësh | JO (internal) | Problem-specifik |
| 13 | Random Seed | 42 | `seed` | PO | Problem-specifik |
| 14 | Lookahead Limit | 6 | `lookahead_limit` | PO | Problem-specifik |
| 15 | Density Percentile | 25% | `density_percentile` | PO | Problem-specifik |
| 16 | Tournament Size (k) | 3 / 5 | `tournament_k` | PO (tani) | GA klasik |
| 17 | Stale Limit | 50 / 60 | `stale_limit` | PO (tani) | GA klasik |
| 18 | Beam Params (window) | max_segs, beam_w, max_exp | Adaptive | JO (internal) | Problem-specifik |
| 19 | Beam Width (seed) | 0–200 | Adaptive | JO (internal) | Problem-specifik |
| 20 | Final Reserve | 4.0s | `_FINAL_RESERVE_S` | JO (konstante) | Problem-specifik |
| 21 | Penalitetet (instance) | switch_pen, term_pen, etc. | Nga inputi | JO | Constraint |
| 22 | Pen Ratio → _fw, _fb | Derivuar | Adaptive | JO | Problem-specifik |
| 23 | Polish Rounds | 2 / 3 | `polish_rounds` | PO (tani) | Problem-specifik |

---

## D. DIAGRAMI I RRJEDHËS

```
┌──────────────────────────────────────────┐
│          generate_solution()             │
│  ┌─────────────────────────────┐         │
│  │ _seed() — 55% e kohës       │         │
│  │  ├─ Beam Search (instanca   │         │
│  │  │  e vogël, param #19)     │         │
│  │  ├─ GRASP ndërtim me        │         │
│  │  │  alpha (#12), strategji  │         │
│  │  └─ → popullatë fillestare │         │
│  └─────────────┬───────────────┘         │
│                │                         │
│  ┌─────────────▼───────────────┐         │
│  │ _postprocess_elite() (#10)  │         │
│  │  destroy–rebuild mbi top-8  │         │
│  └─────────────┬───────────────┘         │
│                │                         │
│  ┌─────────────▼───────────────┐         │
│  │ _ga() — 45% e kohës (#11)   │         │
│  │  while kohë dhe !stagnacion:│         │
│  │    ├─ Elitizëm (#10)        │         │
│  │    ├─ Tournament k=3/5 (#16)│         │
│  │    ├─ Crossover Pc (#2,#6)  │         │
│  │    ├─ Mutacion Pm (#3,#7)   │         │
│  │    └─ Rendit & seleksion    │         │
│  └─────────────┬───────────────┘         │
│                │                         │
│  ┌─────────────▼───────────────┐         │
│  │ _final_polish() (#20)       │         │
│  │  2–3 raunde destroy–rebuild │         │
│  │  në kohën e rezervuar (4s)  │         │
│  └─────────────┬───────────────┘         │
│                │                         │
│                ▼                         │
│         Zgjidhja Finale                  │
└──────────────────────────────────────────┘
```

---

## E. REFERENCAT AKADEMIKE

1. **Eiben, A.E. & Smith, J.E.** (2015). *Introduction to Evolutionary Computing*. Springer. — Parametrat e GA, elitizëm, tournament selection.
2. **De Jong, K.A.** (1975). *Analysis of the behavior of a class of genetic adaptive systems*. PhD Thesis, University of Michigan. — Crossover rate, mutation rate, population size.
3. **Goldberg, D.E. & Deb, K.** (1991). "A comparative analysis of selection schemes used in genetic algorithms." *Foundations of Genetic Algorithms*. — Tournament vs. Roulette Wheel.
4. **Radcliffe, N.J.** (1994). "The algebra of genetic algorithms." *Annals of Mathematics and AI*. — Respectful recombination.
5. **Shaw, P.** (1998). "Using constraint programming and local search methods to solve vehicle routing problems." *CP-98*. — Large Neighborhood Search.
6. **Ropke, S. & Pisinger, D.** (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." *Transportation Science*. — ALNS.
7. **Feo, T.A. & Resende, M.G.C.** (1995). "Greedy randomized adaptive search procedures." *Journal of Global Optimization*. — GRASP, alpha parameter.
8. **Deb, K.** (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley. — Fitness scalarization.
