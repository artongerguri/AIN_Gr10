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


## Përshkrimi i Projektit: Optimizimi i Orarit Televiziv

Ky projekt adreson **Problemin e Planifikimit Televiziv për Hapësira Publike** (TV Channel Scheduling Optimization for Public Spaces) në kuadër të lëndës **Algoritmet e Avancuara**. Objektivi primar është përzgjedhja dhe planifikimi optimal i një nënbashkësie të programeve televizive në kanale të shumta, me qëllim maksimizimin e pikëve totale të shikueshmërisë.

**Kufizimet dhe Qëllimet Kryesore:**

Përveç kufizimeve bazë kohore, problemi përfshin rregulla specifike për të siguruar një përvojë cilësore shikimi:

*   **Time Window Constraint:** Programet duhet të planifikohen strikt brenda intervalit kohor global të përcaktuar (Hapja dhe Mbyllja).
*   **No Overlap Constraint:** Ndalohet rreptësisht mbivendosja kohore e programeve në të njëjtin kanal.
*   **Minimum Duration:** Programet duhet të kenë një kohëzgjatje minimale për t'u konsideruar të vlefshme.
*   **Genre Repetition:** Për të siguruar shumëllojshmëri, ka një kufizim në numrin e programeve të njëpasnjëshme të të njëjtit zhanër.
*   **Priority Blocks:** Blloqe kohore specifike ku vetëm kanale të caktuara kanë prioritet ose lejohen të transmetojnë.
*   **Time Preferences:** Bonuse pikësh për transmetimin e zhanreve të caktuara në orare të preferuara.
*   **Optimization Goal:** Maksimizimi i funksionit objektiv, duke balancuar pikët e programeve me penalitetet e mundshme.

## Beam Search Scheduler

 **Beam Search Scheduler**, është një algoritëm **deterministik** që tejkalon kufizimet e qasjeve standarde **Greedy** përmes eksplorimit paralel të hapësirës së zgjidhjeve.

**Metodologjia:**

1.  **Beam Search Strategy:** Në vend të ndjekjes së një rruge të vetme, algoritmi mirëmban një bashkësi prej $N$ zgjidhjesh të pjesshme më premtuese në çdo hap (**Beam Width**). Kjo mundëson shmangien e minimumeve lokale dhe rikuperimin nga vendimet sub-optimale të hershme.
2.  **Lookahead Mechanism:** Përtej vlerësimit të menjëhershëm, algoritmi implementon një mekanizëm **Lookahead** me thellësi të konfiguruar. Kjo analizon impaktin e vendimeve aktuale në mundësitë e ardhshme, duke parandaluar bllokimin e programeve me vlerë të lartë.
3.  **Density Heuristic:** Për vlerësimin e potencialit të intervaleve kohore të mbetura, përdoret një heuristikë e bazuar në dendësinë e pikëve (pikë/minutë).

**Konfigurimi i Parametrave:**
*   **Beam Width:** 100. Ruan 100 degëzimet më të mira të pemës së kërkimit në çdo iteracion.
*   **Lookahead:** 4 hapa. Vlerëson pasojat e vendimeve deri në 4 nivele thellësi.
*   **Density Percentile:** Fokusohet në 25% të programeve më të mira për vlerësim heuristik)

Ky kombinim i eksplorimit **Beam Search** dhe heuristikave të avancuara **Lookahead** mundëson gjetjen e zgjidhjeve të cilësisë së lartë në mënyrë efikase.

## Ekzekutimi i Projektit

Për të ekzekutuar projektin dhe për të gjeneruar orarin optimal, ndiqni hapat e mëposhtëm:

1.  Sigurohuni që keni të instaluar Python 3.
2.  Hapni terminalin në direktorinë kryesore të projektit.
3.  Ekzekutoni komandën:
    ```bash
    python3 main.py
    ```
4.  Do t'ju shfaqet një listë e fajllave hyrës (input) të disponueshëm (p.sh., `usa_tv_input.json`, `uk_tv_input.json`, etj.).
5.  Shkruani numrin e indeksit që korrespondon me fajllin që dëshironi të procesoni dhe shtypni Enter.

Algoritmi do të fillojë ekzekutimin dhe në fund do të ruajë rezultatin në folderin `data/output/`.

### Zgjedhja e algoritmit

Përveç `Beam Search`, tani mund të përdorni edhe një algoritëm me randomness:

```bash
python3 main.py --algorithm random
```

Algoritëm alternativ me Simulated Annealing:

```bash
python3 main.py --algorithm sa
```

Opsione të dobishme për versionin random:

```bash
python3 main.py --algorithm random --iterations 400 --seed 42
```

```bash
python3 main.py --algorithm sa --iterations 200 --sa-steps 160 --temperature 120 --cooling-rate 0.985 --seed 42
```

* `--algorithm beam|random|sa`: Zgjedh algoritmin
* `--iterations`: Numri i iteracioneve për `random` dhe `sa`
* `--seed`: Seed fiks për rezultate të riprodhueshme për `random` dhe `sa`
* `--temperature`: Temperatura fillestare për `sa`
* `--cooling-rate`: Shpejtësia e ftohjes për `sa`
* `--sa-steps`: Numri i lëvizjeve të fqinjësisë brenda një iteracioni `sa`
=======
# AIN_Gr10
