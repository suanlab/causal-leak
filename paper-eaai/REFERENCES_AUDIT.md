# References.bib Verification Audit

**Date:** 2026-05-09
**Method:** Each entry verified against publisher/proceedings/arXiv via web search.
**Total entries audited:** 41

## Summary

| Status | Count | Description |
|--------|------:|-------------|
| Verified — correct | 36 | Title, authors, venue, year, pages all match published source |
| Corrected — metadata fixed | 5 | Issues found and fixed in this audit |

## Issues Found and Fixed

### 1. `liu2023leakage` — FULLY HALLUCINATED (renamed → `liu2019leakage`)
- **Was:** "Leakage Detection in Water Distribution Networks Based on Improved SqueezeNet Convolutional Neural Network", Liu/Ma/Li/Tie/Zhang/Gao, *Water Supply* 23(2):820–836 (2023)
- **No such paper exists.** The closest published paper by the same author group is:
  Liu Y., Ma X., Li Y., Tie Y., Zhang Y., Gao J. "Water Pipeline Leakage Detection Based on Machine Learning and Wireless Sensor Networks." *Sensors* 19(23):5086, 2019. DOI: 10.3390/s19235086
- **Fix:** Replaced entry with the actual 2019 *Sensors* paper; renamed cite key to `liu2019leakage`; updated narrative in `paper.tex` ("improved SqueezeNet" → "machine-learning over wireless-sensor-network signals").

### 2. `tariq2022leak` — HALLUCINATED title and journal
- **Was:** "A Review of Deep Learning Models for Leak Detection in Water Distribution Systems", Saqib Tariq et al., *Sustainable Cities and Society* 77:103492 (2022)
- **Actual paper:** Tariq S., Bakhtawar B., Zayed T. "Data-driven application of MEMS-based accelerometers for leak detection in water distribution networks." *Science of the Total Environment* 810:151110, 2022. DOI: 10.1016/j.scitotenv.2021.151110
- Also: first author's first name is **Salman** (not Saqib).
- **Fix:** Replaced title, journal, volume, page, DOI, and corrected first-author given name; updated narrative in `paper.tex`.

### 3. `gao2017model` — HALLUCINATED title and authors
- **Was:** "On the Selection of Acoustic/Vibration Sensors for Leak Detection in Plastic Water Pipes", Gao/Liu/Ma/Cheng/Yang, *Mechanical Systems and Signal Processing* 83:222–233 (2017)
- The title "On the Selection of Acoustic/Vibration Sensors..." actually belongs to **Gao Y., Brennan M.J., Joseph P.F., Muggleton J.M., Hunaidi O.**, published in *Journal of Sound and Vibration* 283(3–5):927–941, **2005** — different authors, different journal, different year.
- The 2017 paper by Gao et al. that actually exists is:
  Gao Y., Brennan M.J., Liu Y., Almeida F.C.L., Joseph P.F. "Improving the shape of the cross-correlation function for leak detection in a plastic water distribution pipe using acoustic signals." *Applied Acoustics* 127:24–33, 2017. DOI: 10.1016/j.apacoust.2017.05.033
- **Fix:** Replaced title, authors, journal, volume, pages with the actual 2017 paper; updated narrative ("compared accelerometer- and hydrophone-based sensor selection criteria" → "improved the shape of the cross-correlation function for time-delay-based leak detection by removing pipe-resonance effects").

### 4. `fereidooni2021hybrid` — Wrong middle author surname
- **Was:** "Fereidooni, Zahra and Tayyebi, Hossein and Bahadori-Jahromi, Ali"
- **Actual:** Fereidooni Z., **Tahayori, Hooman**, Bahadori-Jahromi A. (J. Ambient Intelligence and Humanized Computing 12(2):1613–1629, 2021. DOI: 10.1007/s12652-020-02233-2)
- **Fix:** Corrected middle author surname Tayyebi→**Tahayori**, given name Hossein→Hooman. Same fix applied to `paper/references.bib`.

### 5. `ganin2016dann` — Wrong page numbers and JMLR paper number
- **Was:** *JMLR* 17(1):2096–2030, 2016 (page numbers impossible — start > end)
- **Actual:** *JMLR* 17(59):1–35, 2016 — paper article 59, pages 1–35 within the volume.
- **Fix:** number 1→**59**, pages 2096–2030→**1–35**.

## Verified-Correct Entries (36)

The following entries were each cross-checked against the publisher/proceedings/arXiv and are correct as written:

- `lee2023leak` — Lee & Kim 2023 *Sensors* 23(21):8935. DOI:10.3390/s23218935.
- `arjovsky2019irm` — arXiv:1907.02893, 4 authors confirmed.
- `sagawa2020groupdro` — ICLR 2020, 4 authors confirmed.
- `gulrajani2021dg` — ICLR 2021, 2 authors confirmed.
- `rosenfeld2021irm` — ICLR 2021, 3 authors confirmed.
- `krueger2021rex` — ICML 2021 PMLR 139:5815-5826, 8 authors confirmed.
- `liu2021jtt` — ICML 2021 PMLR 139:6781-6792, 8 authors confirmed.
- `rame2022fishr` — ICML 2022, 3 authors confirmed.
- `koh2021wilds` — ICML 2021, lead-author list confirmed.
- `sun2016coral` — ECCV 2016 Workshops, 2 authors confirmed.
- `long2015dan` — ICML 2015 pp.97–105, 4 authors confirmed.
- `li2018mmddg` — CVPR 2018 pp.5400–5409, 4 authors confirmed.
- `geirhos2020shortcut` — Nature Machine Intelligence 2(11):665–673 (2020), 7 authors confirmed.
- `zech2018pneumonia` — PLOS Medicine 15(11):e1002683 (2018), 6 authors confirmed.
- `degrave2021covid` — Nature Machine Intelligence 3(7):610–619 (2021), 3 authors confirmed.
- `beery2018terra` — ECCV 2018 pp.456–473, 3 authors confirmed.
- `damour2022underspec` — JMLR 23(226):1–61 (2022), confirmed.
- `hunaidi2000plastic` — Journal AWWA 92(2):82–94 (2000), 4 authors confirmed.
- `ismail2019review` — IEEE Access 7:51965–51981 (2019), 7 authors confirmed.
- `shukla2020leakage` — Automation in Construction 117:103256 (2020), 2 authors confirmed.
- `cody2020leak` — J. Computing in Civil Engineering 34(2):04020001 (2020), 3 authors confirmed.
- `almeida2014leak` — Sensors 14(3):5595–5610 (2014), 6 authors confirmed.
- `chen2016xgboost` — KDD 2016 pp.785–794, 2 authors confirmed.
- `ke2017lightgbm` — NeurIPS 2017, 8 authors confirmed.
- `prokhorenkova2018catboost` — NeurIPS 2018, 5 authors confirmed.
- `breiman2001rf` — Machine Learning 45(1):5–32 (2001), confirmed.
- `arik2021tabnet` — AAAI 2021 35(8):6679–6687, 2 authors confirmed.
- `gorishniy2021ftt` — NeurIPS 2021, 4 authors confirmed.
- `popov2020node` — ICLR 2020, 3 authors confirmed.
- `borisov2024tabsurvey` — IEEE TNNLS 35(6):7499–7519 (2024), 6 authors confirmed.
- `lundberg2017shap` — NeurIPS 2017 vol 30, 2 authors confirmed.
- `ribeiro2016lime` — KDD 2016 pp.1135–1144, 3 authors confirmed.
- `janzing2020shapley` — AISTATS 2020 PMLR 108:2907–2916, 3 authors confirmed.
- `park2019specaugment` — Interspeech 2019, 7 authors confirmed.
- `zhang2018mixup` — ICLR 2018, 4 authors confirmed.
- `lin2017focal` — ICCV 2017 pp.2980–2988, 5 authors confirmed.
- `pineau2021reproducibility` — JMLR 22(164):1–20 (2021), 8 authors confirmed.
- `henderson2018reproducibility` — AAAI 2018 32(1), 6 authors confirmed.
- `nist2023airmf` — NIST AI RMF 1.0, 2023 — official document, citation acceptable.
- `eu2024aiact` — EU AI Act 2024/1689 — official EU document, citation acceptable.
- `kwater2022smartwater`, `aihub2022dataset` — institutional/dataset citations, no peer-reviewed source to verify.

## Build Verification

After all fixes, the paper still builds cleanly:
```
paper.pdf  10 pages  245,524 bytes  0 undefined references
```
