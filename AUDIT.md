# Final Integrated Audit

## 1. Executive verdict

이 저장소는 현재 상태로는 serious AI/ML venue 제출용 최종본으로 보기 어렵다. 핵심 분류 실험의 상당 부분은 코드와 결과 아티팩트로 추적 가능하고, `macro-F1`/`accuracy` 자체의 기본 구현은 대체로 정상이다. 그러나 제출 직전 게이트 기준에서는 세 가지가 치명적이다. 첫째, 현재 제출 타깃으로 보이는 `paper-eaai/` 파이프라인이 저장소의 공식 재현 경로와 분리되어 있다. `README.md`, `Makefile`, `scripts/reproduce_all.sh`, `scripts/verify.sh`는 여전히 구 논문 `paper/` 중심이고, `REPLICATION.md`의 “single command reproduces the entire pipeline” 주장도 실제 코드와 다르다. 둘째, 운영 metric인 detection-rate-at-FAR는 테스트 셋에서 threshold를 직접 맞춰 계산하고 있으며, 심지어 `1% FAR` 목표를 실제로 달성하지도 못한다. 셋째, 비용/파레토 근거는 실제 측정값과 합성 proxy가 섞여 있고, 일부 논문 문장은 존재하지 않는 multi-seed 또는 비용 실험을 암시한다.

재현성은 `Partial`이 아니라 사실상 제출 관점에서는 `No`에 가깝다. 핵심 실험 스크립트는 import/compile/load smoke test를 통과했지만, 현재 저장소의 공식 진입점만으로 EAAI 원고 전체를 재생성하는 경로는 없다. Ablation은 초기 핵심 실험(E1 이전의 matrix/LORO/causal-band/deep baseline/PGA/ensemble)은 비교적 잘 남아 있으나, E3/E4/E5/E6/E7은 공정성 또는 metric 타당성 면에서 약하다. Metric 신뢰도는 core macro-F1은 높지만 운영 metric과 Pareto/cost 주장은 신뢰도가 낮다.

상위 3개 제출 차단 요인:
1. `paper-eaai` 제출 경로와 공식 재현 경로의 구조적 불일치
2. 테스트 셋 기반 threshold tuning으로 계산된 E5 운영 metric
3. 합성 ensemble 비용점과 불일치한 baseline 설정에 의존한 cost/Pareto 주장

## 2. Submission blockers

### Severity: Blocker
**Title**  
현재 제출 타깃(`paper-eaai`)이 저장소의 공식 재현/검증 경로로 재생성되지 않는다

**Why this blocks submission**  
리뷰어가 저장소의 공식 문서와 엔트리포인트를 따라가면 EAAI 원고가 아니라 구 버전 `paper/` 파이프라인을 재생성하게 된다. 제출용 원고와 코드의 canonical path가 둘로 갈라져 있으면, “which paper does this repository reproduce?”라는 가장 기본 질문에 답하지 못한다.

**Exact evidence**  
- `README.md:30,74-82,109-135`는 전체 write-up과 재현 경로를 `paper/paper.tex`, `scripts/reproduce_all.sh`, `make paper`, `make verify`로 안내한다.
- `scripts/reproduce_all.sh:31-58`는 구 파이프라인만 실행하고 마지막에 `cd paper && pdflatex ... paper.tex`를 돌린다. `paper-eaai/`는 전혀 다루지 않는다.
- `Makefile:7-24` 역시 `scripts/reproduce_all.sh`와 `paper/`만 호출한다.
- `scripts/verify.sh:19-35`는 `paper/paper.log`, `paper/paper.pdf`, `paper/paper.tex`만 검사한다.
- 반면 제출 타깃으로 보이는 EAAI 원고는 `paper-eaai/paper.tex`이며, 여기에만 `fig9`~`fig14`, E1~E7 결과가 들어간다 (`paper-eaai/paper.tex:903,967,1029,1083,1095` 등).
- `REPLICATION.md:59-83`는 “A single shell command reproduces the entire results pipeline”라고 하지만 실제로는 `bash src/run_multiseed.sh` 후 E2~E7을 별도 수동 실행해야 한다.
- `REPLICATION.md:90-96`의 figure regeneration 명령은 `fig1`~`fig8`만 포함하고, EAAI 추가 그림 `fig9`~`fig14`는 누락되어 있다.

**Risk to conclusions**  
논문-코드 정합성이 무너진다. 리뷰어 또는 AC가 원고의 어떤 표/그림/수치를 어떤 명령으로 재생성해야 하는지 명확히 추적할 수 없다.

**Minimum fix**  
하나의 canonical submission target만 남기고, `README.md`/`REPLICATION.md`/`Makefile`/`scripts/reproduce_all.sh`/`scripts/verify.sh`를 모두 `paper-eaai/` 기준으로 통일해야 한다. E1~E7 전체와 `paper-eaai` PDF 빌드를 한 경로로 묶어야 한다.

**Verification after fix**  
클린 환경에서 문서 한 곳에 적힌 단일 명령만으로 EAAI figures/tables/PDF가 생성되는지 확인하고, `scripts/verify.sh`가 `paper-eaai`를 검사하도록 바뀌었는지 확인.

### Severity: Blocker
**Title**  
E5 운영 metric이 테스트 셋에서 threshold를 직접 맞춰 계산되어 평가 프로토콜이 무효다

**Why this blocks submission**  
운영 metric은 deployment claim의 핵심인데, 현재 구현은 테스트 셋 negative 분포를 보고 threshold를 고른 뒤 같은 테스트 셋에서 detection rate를 보고한다. 이는 명백한 test-aware tuning이다. 더 나쁘게는 논문이 말하는 `1% FAR`도 실제로 만족하지 못한다.

**Exact evidence**  
- `src/run_cost_metrics.py:6-11`은 “find the operating point with false-alarm rate <= 1% on the test set”라고 명시한다.
- `src/run_cost_metrics.py:92-105`의 `detection_rate_at_far`는 `neg = prob_leak[y_true_binary == 0]` 후 `np.quantile(neg, 1.0 - far_target)`로 threshold를 직접 테스트 셋에서 산출한다.
- 동일 스크립트는 이 threshold로 같은 테스트 셋의 detection rate를 계산한다 (`src/run_cost_metrics.py:101-105`).
- 실제 산출물 `results/cost_metrics_results.json:1-37`에서 `far_0.01`의 `achieved_far`는 flat/hier 모두 `0.022222...`로, 목표 1%가 아니라 2.22%다.
- 논문은 이 결과를 운영 claim으로 끌어온다 (`paper-eaai/paper.tex:985-992`, `1112-1117`).

**Risk to conclusions**  
“deployment recipe achieves 100% detection at 2.22% FAR”라는 운영 해석은 out-of-sample 성능이 아니라 test-tuned operating point다. 실서비스 임계값 설계나 utility-side false alarm budgeting 주장을 정당화할 수 없다.

**Minimum fix**  
threshold는 training/validation only로 정해야 한다. 예를 들어 train-derived calibration split 또는 source-domain validation에서 threshold를 고정하고, test는 한 번만 평가해야 한다. “1% FAR”을 못 맞추면 그대로 보고해야 한다.

**Verification after fix**  
새 산출물에서 threshold derivation source가 test가 아닌지 확인하고, 독립 test에서 `achieved_far`와 `detection_rate`를 재계산해 논문 문장과 일치하는지 확인.

### Severity: Blocker
**Title**  
비용/Pareto 근거가 합성 proxy와 불일치한 설정에 의존해 submission-quality evidence가 아니다

**Why this blocks submission**  
논문은 deployment recipe의 cost-vs-accuracy/Pareto 위치를 강하게 주장하지만, 실제 figure code는 ensemble 점을 직접 측정하지 않고 proxy로 합성한다. 게다가 cost benchmark와 extra baseline benchmark가 서로 다른 하이퍼파라미터를 사용한다. 이런 상태에서는 비용 비교가 공정하거나 재현 가능하다고 보기 어렵다.

**Exact evidence**  
- `src/make_fig14_pareto.py:63-76`는 ensemble 점을 “special anchor point”로 추가하면서 inference cost를 `2 * xgboost_flat`로 근사하고, RAM도 `xgboost_flat + xgboost_hier` 합으로 근사한다.
- 같은 코드 `src/make_fig14_pareto.py:72`는 존재하지 않는 key `tree_soft_average_f1`를 읽고, 없으면 기본값 `0.4326`을 사용한다.
- 실제 `results/ensemble_results.json:2-36`에는 `tree_soft_average_f1`가 없고 `ensembles.tree_only_soft_avg.f1`만 존재한다.
- 논문은 이 합성 점을 실제 Pareto frontier claim으로 사용한다 (`paper-eaai/paper.tex:1096-1101`).
- `src/run_extra_baselines.py:93-118`는 LightGBM/CatBoost를 `200 trees / lr 0.1`으로 돌린다.
- `src/run_compute_cost.py:189-212`는 같은 모델들을 `400 trees / lr 0.05`로 다시 측정한다.
- 논문은 cost benchmark를 baseline 비교와 연결해 해석한다 (`paper-eaai/paper.tex:1042-1058`).

**Risk to conclusions**  
“Pareto frontier”와 “운영 비용 대비 우월성”이 직접 측정된 실험 결과가 아니라, 일부는 근사치이고 일부는 다른 설정에서 얻은 숫자다. 비용-정확도 tradeoff 주장이 과장되거나 잘못될 수 있다.

**Minimum fix**  
ensemble도 실제 추론 경로로 benchmark해야 한다. cost benchmark는 accuracy benchmark와 동일 모델/동일 하이퍼파라미터/동일 선택 규칙을 사용해야 한다. Fig. 14는 실제 측정치만 사용해야 한다.

**Verification after fix**  
`results/compute_cost_results.json`에 ensemble 및 비교 대상이 실제 측정값으로 들어가고, figure code가 fallback 없이 해당 key를 직접 읽는지 확인.

## 3. Critical findings

### Severity: Critical
**Title**  
논문이 존재하지 않는 “multi-seed LightGBM comparison”을 암시한다

**Why this matters for scientific validity**  
single-seed 우연을 논문 스스로 경계해야 하는데, 실제로는 multi-seed가 없는 모델에 대해 “within noise”라는 해석을 붙이고 있다. 이는 variance evidence를 가장한 unsupported prose다.

**Exact evidence**  
- `paper-eaai/paper.tex:932-935`는 “The single-seed lead of LightGBM ... the multi-seed comparison shrinks the gap to within noise”라고 서술한다.
- `src/run_extra_baselines.py:33-35`는 seed를 고정 `42`로 두고, 출력도 `results/extra_baselines_results.json` 하나만 쓴다.
- `results/extra_baselines_results.json` 역시 단일 `seed: 42`만 가진다.
- `src/aggregate_multiseed.py:150-156`는 MLP/CNN/augmentation/PGA만 집계하며 LightGBM은 포함하지 않는다.

**Failure mode or bias introduced**  
single-seed favorable outcome를 “noise 수준”이라고 해석하면서 baseline lead를 과도하게 약화시킨다.

**Minimum fix**  
LightGBM/XGBoost/CatBoost 등 tree baselines도 같은 seed protocol로 multi-seed 반복을 하거나, 현재 원고에서 “within noise” 문구를 삭제해야 한다.

**Verification after fix**  
multi-seed baseline summary가 새로 생성되거나, 문장이 single-seed limitation을 명시하는 방향으로 수정되었는지 확인.

### Severity: Critical
**Title**  
“7 additional baselines의 비용 측정”이라는 claim이 구현과 맞지 않는다

**Why this matters for scientific validity**  
논문 contribution이 cost analysis coverage를 과장한다. 비용이 측정되지 않은 모델까지 측정된 것처럼 쓰면 empirical support 범위를 넘는다.

**Exact evidence**  
- `paper-eaai/paper.tex:174-180`는 LightGBM, CatBoost, TabNet, RandomForest, DeepCORAL, DANN, MMD-DG에 대해 training time / inference latency / RAM / GPU memory를 “for each” 측정한다고 읽힌다.
- 실제 `src/run_compute_cost.py:353-366`의 benchmark 대상은 `xgboost_flat`, `xgboost_hier`, `lightgbm`, `catboost`, `random_forest`, `mlp_erm`, `cnn_erm`뿐이다.
- TabNet, DeepCORAL, DANN, MMD-DG, ensemble은 비용 benchmark에 없다.

**Failure mode or bias introduced**  
cost coverage가 실제보다 넓어 보이며, 독자가 “모든 추가 baseline이 비용까지 검증되었다”고 오해한다.

**Minimum fix**  
논문 문장을 실제 benchmark 범위에 맞추거나, 빠진 모델들의 비용 측정을 추가해야 한다.

**Verification after fix**  
원고의 대상 모델 리스트와 `compute_cost_results.json`의 모델 목록이 일치하는지 확인.

### Severity: Critical
**Title**  
domain-alignment baseline 비교가 “only auxiliary loss differs”라는 설명과 달리 confounded되어 있다

**Why this matters for scientific validity**  
ablation이나 baseline fairness에서 training budget이 다르면 auxiliary loss의 효과를 깨끗하게 해석할 수 없다.

**Exact evidence**  
- `src/run_domain_alignment.py:13-16`는 “the only difference ... is the auxiliary loss term. This makes the comparison clean.”라고 주장한다.
- 그러나 DA config는 `epochs=80`, `early_stop_patience=15` (`src/run_domain_alignment.py:145-155`, `293-298`)다.
- 대응 MLP baseline은 `epochs=120`, `early_stop_patience=25` (`src/run_irm_mlp.py:57-60`)다.
- 즉 동일 backbone/sampler라도 stopping budget과 optimization horizon이 다르다.

**Failure mode or bias introduced**  
DA 계열이 불리한 budget을 받거나 다른 selection pressure를 받아 baseline 비교가 혼탁해진다.

**Minimum fix**  
MLP ERM/IRM/GroupDRO와 DA methods의 epochs, patience, selection rule, batch size, weight decay를 동일하게 맞추고 다시 비교해야 한다.

**Verification after fix**  
config diff를 문서화하고, 코드 상 parameter set이 실제로 동일한지 line-by-line 비교.

## 4. Major findings

### Severity: Major
**Title**  
`run_multiseed.sh`가 clean reproduction을 보장하지 않고 partial failure를 숨길 수 있다

**Why this matters for scientific validity**  
multi-seed summary는 variance claims의 근거다. seed가 빠지거나 실패해도 summary가 생성되면 mean/std와 p-value가 silently 왜곡될 수 있다.

**Exact evidence**  
- `src/run_multiseed.sh:11-13`은 `set -uo pipefail`만 사용하고 실패를 계속 진행하도록 주석으로 명시한다.
- `src/run_multiseed.sh:15-20`은 `SEEDS=(0 1 2 3)`만 실행하고 seed 42는 “results exist already”라고 가정한다.
- `src/run_multiseed.sh:20`은 절대경로 `PYBIN="${PYBIN:-/home/suan/Projects/Water/.venv/bin/python}"`를 기본값으로 박아 둔다.
- `src/aggregate_multiseed.py:41-44`는 존재하는 파일만 읽고, 누락 seed가 있어도 에러를 내지 않는다.

**Failure mode or bias introduced**  
5-seed라고 보고하면서 실제로는 3-seed 또는 4-seed만 집계될 수 있다. 절대경로 기본값은 타 머신 재현을 깨뜨린다.

**Minimum fix**  
seed 42까지 포함한 완전 실행으로 바꾸고, 누락 파일이 있으면 hard fail하도록 해야 한다. 기본 interpreter는 상대경로나 `python3` 탐지로 바꿔야 한다.

**Verification after fix**  
없는 seed artifact 상태에서 스크립트가 실패하는지, 새로 돌렸을 때 정확히 5개 seed 파일이 생기는지 확인.

### Severity: Major
**Title**  
기존 verification gate가 현재 제출 원고를 전혀 검증하지 않는다

**Why this matters for scientific validity**  
제출 직전 품질 게이트가 구 논문만 검사하면, 최신 원고의 broken reference나 stale figure가 그대로 통과한다.

**Exact evidence**  
- `scripts/verify.sh:19-35`는 `paper/paper.log`, `paper/paper.pdf`, `paper/paper.tex`만 검사한다.
- `Makefile:20-24`의 `paper`/`verify` 타깃도 모두 `paper/`만 대상으로 한다.
- `paper-eaai/paper.tex`가 현재 확장 원고인데 이에 대한 검증 경로는 없다.

**Failure mode or bias introduced**  
제출 PDF에서 깨진 citation, 잘못된 그림, 구 수치 잔존이 생겨도 자동으로 잡히지 않는다.

**Minimum fix**  
verification 대상과 build target을 `paper-eaai`로 옮기고, EAAI figures 9~14 생성 여부도 체크해야 한다.

**Verification after fix**  
`scripts/verify.sh`가 `paper-eaai/paper.log`와 `paper-eaai/paper.pdf`를 검사하는지 확인.

### Severity: Major
**Title**  
`run_cost_metrics.py`의 설명과 실제 동작이 다르다

**Why this matters for scientific validity**  
운영 metric이 “post-process only”인지 “재학습 포함”인지에 따라 해석과 재현 비용이 달라진다. 문서가 다르면 독자가 같은 숫자를 재현하기 어렵다.

**Exact evidence**  
- docstring은 “This script post-processes already-computed predictions, so it does not retrain anything.” (`src/run_cost_metrics.py:19-20`)라고 적는다.
- 실제 본문은 flat XGB와 hierarchical를 다시 학습한다 (`src/run_cost_metrics.py:118-119`, `133-135`).

**Failure mode or bias introduced**  
독자가 기존 prediction artifact만으로 E5/E7이 계산된다고 오해한다.

**Minimum fix**  
문서를 실제 동작에 맞게 수정하거나, 진짜 post-hoc 계산으로 재구성해야 한다.

**Verification after fix**  
docstring/README/REPLICATION 설명과 호출 그래프가 일치하는지 확인.

### Severity: Major
**Title**  
cost benchmark가 accuracy benchmark와 다른 하이퍼파라미터를 사용한다

**Why this matters for scientific validity**  
비용-정확도 비교는 동일 모델 정의를 전제로 해야 한다. 다른 tree count / learning rate로 얻은 비용은 accuracy 결과와 직접 비교할 수 없다.

**Exact evidence**  
- `src/run_extra_baselines.py:93-118`  
  - LightGBM: `n_estimators=200`, `learning_rate=0.1`
  - CatBoost: `iterations=200`, `learning_rate=0.1`
- `src/run_compute_cost.py:189-212`  
  - LightGBM: `n_estimators=400`, `learning_rate=0.05`
  - CatBoost: `iterations=400`, `learning_rate=0.05`

**Failure mode or bias introduced**  
비용은 더 큰 모델에서, 정확도는 다른 모델에서 나온다. 특정 baseline이 과도하게 비싸거나 느리게 보일 수 있다.

**Minimum fix**  
동일 하이퍼파라미터로 cost/accuracy를 맞추거나, “cost-only tuned variant”임을 명시적으로 분리해야 한다.

**Verification after fix**  
비용 스크립트와 정확도 스크립트의 model config가 동일한지 diff 검사.

### Severity: Major
**Title**  
LightGBM 비용 수치가 통제된 benchmark가 아니라 contention artifact임을 논문이 스스로 인정한다

**Why this matters for scientific validity**  
비용 benchmark는 controlled measurement여야 한다. “shared machine contention” 값은 모델 고유 비용이 아니다.

**Exact evidence**  
- `paper-eaai/paper.tex:1049-1054`는 table의 `2944 s`가 “shared machine ... contention with the multi-seed sweep running concurrently”를 반영한다고 적는다.
- `REPLICATION.md:119-120`도 LightGBM CPU contention 문제를 언급한다.
- 그런데 같은 cost table은 main result evidence로 사용된다 (`paper-eaai/paper.tex:1060-1077`).

**Failure mode or bias introduced**  
LightGBM이 비현실적으로 느린 모델처럼 보일 수 있다.

**Minimum fix**  
isolated machine 또는 at least isolated process 조건에서 다시 측정하고, contention run은 폐기해야 한다.

**Verification after fix**  
독립 반복 측정에서 train time 분산을 함께 보고, contention이 제거되었는지 확인.

### Severity: Major
**Title**  
Fig. 14는 실제 benchmark가 아니라 synthetic ensemble anchor를 그린다

**Why this matters for scientific validity**  
figure가 claim의 핵심 증거라면 synthetic point는 명시적으로 표시되어야 한다. 현재는 실측 Pareto frontier처럼 보인다.

**Exact evidence**  
- `src/make_fig14_pareto.py:63-76`는 ensemble을 actual benchmark가 아니라 approximation으로 추가한다.
- 논문 caption은 이를 “Pareto frontier”라고 직접 말한다 (`paper-eaai/paper.tex:1096-1101`).

**Failure mode or bias introduced**  
독자가 ensemble이 실제로 그 좌표에서 측정되었다고 오해한다.

**Minimum fix**  
실측이면 실측으로 바꾸고, 아니면 figure/caption에 “synthetic aggregate estimate”라고 명시해야 한다.

**Verification after fix**  
figure code가 approximation branch 없이 measured artifact만 쓰는지 확인.

## 5. Moderate findings

### Severity: Moderate
**Title**  
seed control은 존재하지만 완전한 determinism 설계는 아니다

**Why this matters for scientific validity**  
seed 고정이 partial이면 exact rerun drift가 생길 수 있다. multi-seed 연구에서는 치명도는 낮지만 reproducibility score를 깎는다.

**Exact evidence**  
- `src/irm_core.py:43-50`는 NumPy/Torch seed만 설정한다.
- Python `random.seed`, `torch.backends.cudnn.deterministic`, `torch.backends.cudnn.benchmark` 제어는 없다.
- tree baselines/TabNet/CatBoost/LightGBM은 일부 seed 인자를 주지만, 라이브러리별 완전 deterministic 설정 문서는 없다.

**Failure mode or bias introduced**  
GPU/라이브러리 버전에 따라 exact score drift 가능.

**Minimum fix**  
seed policy를 문서화하고, 가능한 deterministic flag를 추가하며, exact reproducibility 한계를 명시.

**Verification after fix**  
동일 seed 재실행 2회에서 score drift가 없는지 또는 허용 범위를 문서화했는지 확인.

### Severity: Moderate
**Title**  
데이터 경로가 저장소 바깥에 하드코딩되어 있다

**Why this matters for scientific validity**  
외부 dataset 자체는 불가피하지만, 경로 가정이 강하면 재현성 문서 없이는 실행이 막힌다.

**Exact evidence**  
- `src/data.py:6`는 `../../kcc-water/Dataset`를 고정한다.
- `README.md:116-118`, `REPLICATION.md:46-55`가 같은 layout을 요구한다.

**Failure mode or bias introduced**  
타 사용자가 동일 상대경로를 만들지 않으면 모든 스크립트가 바로 실패한다.

**Minimum fix**  
환경변수 또는 CLI argument로 data root를 받을 수 있게 해야 한다.

**Verification after fix**  
임의 경로 데이터 루트에서 동일 스크립트가 동작하는지 확인.

### Severity: Moderate
**Title**  
`paper-eaai/claims.csv`가 최신 결과와 동기화되지 않았다

**Why this matters for scientific validity**  
claim tracking 파일이 stale하면 어떤 숫자가 최종 근거인지 내부적으로도 흔들린다.

**Exact evidence**  
- `paper-eaai/claims.csv:52-62`는 E1~E7 claim을 여전히 `pending`으로 둔다.
- 그러나 대응 산출물은 이미 존재한다: `results/multiseed_summary.json`, `results/sigungu_ood_results.json`, `results/extra_baselines_results.json`, `results/domain_alignment_results.json`, `results/cost_metrics_results.json`, `results/compute_cost_results.json`.

**Failure mode or bias introduced**  
내부 검증 체계와 최종 원고가 어긋난다.

**Minimum fix**  
claim tracker를 최신 수치로 업데이트하고 원고 수치와 cross-check해야 한다.

**Verification after fix**  
모든 `pending` claim이 artifact 경로와 함께 `verified` 또는 명시적 `dropped` 상태가 되는지 확인.

### Severity: Moderate
**Title**  
시군구 LOCO 결과는 유효하지만 36개 전체 환경을 대표한다고 일반화하기 어렵다

**Why this matters for scientific validity**  
county-level extension의 설득력은 coverage에 달려 있다.

**Exact evidence**  
- `src/run_sigungu_ood.py:7-8,34,61-63`는 `>=4 classes` 조건으로 16/36 fold만 사용한다.
- 논문도 이를 적는다 (`paper-eaai/paper.tex:152-153`, `699-700`).

**Failure mode or bias introduced**  
county-level robustness claim이 “all counties”가 아니라 “eligible counties”에만 해당한다.

**Minimum fix**  
원고 전체에서 이 한계를 더 일관되게 강조해야 한다.

**Verification after fix**  
abstract/contribution/caption/section text 전부가 “eligible 16/36”를 명시하는지 확인.

### Severity: Moderate
**Title**  
통계 검정은 일부 pair에만 있고, tree baseline variance는 실질적으로 비검증 상태다

**Why this matters for scientific validity**  
variance claim을 selective하게만 제시하면 유리한 모델만 안정적으로 보일 수 있다.

**Exact evidence**  
- `src/aggregate_multiseed.py:158-183`는 Wilcoxon을 MLP GroupDRO vs ERM, IRM vs ERM 두 pair에만 계산한다.
- tree baselines, LightGBM, CatBoost, ensemble에는 multi-seed uncertainty가 없다.

**Failure mode or bias introduced**  
일부 비교는 variance-aware, 일부는 single-seed로 남아 표 해석이 비대칭적이다.

**Minimum fix**  
headline 비교군 전부에 대해 동일한 variance reporting 정책을 적용해야 한다.

**Verification after fix**  
최종 표의 모든 주요 모델에 seed count와 dispersion 정보가 있는지 확인.

## 6. Minor findings

### Severity: Minor
**Title**  
`run_compute_cost.py` docstring의 benchmark 목록과 실제 목록이 다르다

**Why this matters for scientific validity**  
문서 drift는 비용 실험 해석을 혼란스럽게 한다.

**Exact evidence**  
- docstring은 `MLP GroupDRO`를 benchmark 대상에 포함한다 (`src/run_compute_cost.py:13-15`).
- 실제 benchmark append 목록에는 `mlp_erm`만 있고 `groupdro`는 없다 (`src/run_compute_cost.py:353-366`).

**Failure mode or bias introduced**  
독자가 strongest deep baseline의 cost가 측정된 것으로 오해할 수 있다.

**Minimum fix**  
docstring을 수정하거나 실제로 GroupDRO도 benchmark에 포함.

**Verification after fix**  
docstring과 benchmark list 일치 여부 확인.

### Severity: Minor
**Title**  
EAAI PDF 빌드 로그에 hyperref Unicode warning이 남아 있다

**Why this matters for scientific validity**  
과학적 결론을 깨지는 않지만, 제출 품질 관리상 마무리가 덜 된 신호다.

**Exact evidence**  
- `paper-eaai/paper.log`에 `Package hyperref Warning: Token not allowed in a PDF string (Unicode)`가 여러 번 나온다 (예: lines `1235, 1239, 1243, 1247, 1278...`).

**Failure mode or bias introduced**  
PDF metadata/bookmark 품질 저하 가능.

**Minimum fix**  
section title/caption의 수식/Unicode 처리 정리.

**Verification after fix**  
EAAI log에서 해당 warning 소거 확인.

### Severity: Minor
**Title**  
저장소에 stale audit 문서가 다수 공존한다

**Why this matters for scientific validity**  
최종 진실의 원천이 흐려진다.

**Exact evidence**  
- 루트에 `AUDIT_REPORT.md`, `INTEGRITY_AUDIT.md`, `PRESUBMISSION_AUDIT.md`가 있고, `paper-eaai/`에도 `INTEGRITY_AUDIT.md`, `EAAI_AUDIT.md`, `REFERENCES_AUDIT.md`가 있다.

**Failure mode or bias introduced**  
리뷰어가 어떤 감사 결과가 최신인지 알기 어렵다.

**Minimum fix**  
최신 audit 하나만 canonical로 남기고 나머지는 archive 처리.

**Verification after fix**  
문서 링크가 하나의 최신 audit만 가리키는지 확인.

## 7. Reproducibility scorecard

| Item | Verdict | Notes |
|---|---|---|
| environment recreation | Partial | Python deps는 존재하나 LaTeX/toolchain/data path/clean-room EAAI 경로가 완성되지 않음 |
| dependency pinning | Partial | `requirements.txt`는 pin, `requirements-eaai.txt`는 range pin만 사용 |
| seed control | Partial | NumPy/Torch seed는 있으나 full deterministic policy는 없음 |
| determinism awareness | Partial | split seed는 고정, GPU/library determinism 문서화 부족 |
| train/eval config consistency | Fail | extra baseline vs cost benchmark hyperparameter drift, DA budget drift |
| checkpoint reproducibility | Partial | deep model selection logic는 명시돼 있으나 multiseed는 preexisting seed42 artifact에 의존 |
| command reproducibility | Fail | 공식 엔트리포인트가 `paper/`와 `paper-eaai/` 사이에서 갈라짐 |
| dataset split traceability | Pass | 핵심 split logic는 코드로 추적 가능 (`region 43` holdout, fixed split seed) |
| artifact traceability | Partial | core results는 추적 가능하지만 EAAI claim tracker/verification path가 stale |
| notebook/script consistency | Pass | `.ipynb` 파일은 발견되지 않았고 주요 결과는 script 기반 |

## 8. Ablation integrity scorecard

| Name | Claimed purpose | Exists in code? | Actually isolated? | Fairly controlled? | Result traceable? | Verdict |
|---|---|---|---|---|---|---|
| Four-probe shortcut audit | 원 논문 accuracy의 shortcut 원인 진단 | 예 | 예 | 예 | 예 | Verified |
| Row/GroupSite/LORO split audit | OOD 붕괴 정량화 | 예 | 예 | 예 | 예 | Verified |
| Causal-band retraining | 121–422 Hz causal subset 검증 | 예 | 대체로 예 | 대체로 예 | 예 | Verified |
| MLP/CNN ERM vs IRM vs GroupDRO | loss-level invariance 비교 | 예 | 예 | 부분 | 예 | Verified |
| Multi-seed augmentation sweep | CNN recipe 비교 | 예 | 예 | 부분 | 예 | Verified |
| PGA sweep | soft-prior negative result 검증 | 예 | 예 | 부분 | 예 | Verified |
| Tree-only ensemble | deployment candidate | 예 | 예 | 예 | 예 | Verified |
| Sigungu LOCO | county-level OOD extension | 예 | 예 | 부분 | 예 | Verified |
| Extra baselines (RF/LGBM/CatBoost/TabNet) | stronger tabular baselines 비교 | 예 | 부분 | 아니오 | 예 | Confounded |
| Domain alignment (CORAL/DANN/MMD) | DA family 비교 | 예 | 부분 | 아니오 | 예 | Confounded |
| E5 detection-rate@FAR | 운영 detection metric | 예 | 아니오 | 아니오 | 예 | Invalid |
| E7 threshold sweep | stage-1 threshold sensitivity | 예 | 예 | 부분 | 예 | Weak |
| E6 cost benchmark | deployment cost 비교 | 예 | 부분 | 아니오 | 예 | Weak |

## 9. Metric integrity scorecard

| Metric | Where implemented | Where reported | Verification status | Main concern if any |
|---|---|---|---|---|
| 5-class macro-F1 / accuracy | `src/irm_core.py:263-289`, sklearn metric calls across scripts | `paper-eaai/paper.tex`, multiple `results/*.json` | Verified | 기본 구현은 타당 |
| Per-class F1 | same + per-script sklearn calls | tables/figures/results JSON | Verified | 없음 |
| Multi-seed mean/std | `src/aggregate_multiseed.py:47-135` | `results/multiseed_summary.{json,md}`, `paper-eaai/paper.tex:831-847` | Partial | 일부 모델 family만 집계, missing seed silent 허용 |
| Paired Wilcoxon p-values | `src/aggregate_multiseed.py:138-183` | `paper-eaai/paper.tex:836-839` | Partial | 두 pair에만 적용 |
| Detection-rate@FAR | `src/run_cost_metrics.py:92-105` | `paper-eaai/paper.tex:985-992,1112-1117` | Suspicious | test-set threshold tuning, target FAR 미달 |
| Threshold sweep E7 | `src/run_cost_metrics.py:148-188` | `paper-eaai/paper.tex:988-992,1029-1037` | Partial | same test fold에서 post-hoc operational interpretation |
| Compute cost metrics | `src/run_compute_cost.py:91-140` | `paper-eaai/paper.tex:1042-1090` | Partial | benchmark 설정 drift, contention artifact |
| Pareto frontier | `src/make_fig14_pareto.py:63-76` | `paper-eaai/paper.tex:1096-1101` | Suspicious | synthetic ensemble point + fallback constant |

## 10. Claim-to-evidence matrix

| Claim | Support status | Evidence | Main risk / gap |
|---|---|---|---|
| Identifier lookup reproduces ~99.77% published accuracy | Supported | `results/matrix_results.json`, `README.md:34-38`, `paper-eaai/paper.tex:141-147` | 없음 |
| Under LORO, original XGBoost collapses to 56.8% acc / 0.304 F1 | Supported | `results/loro_results.json`, `paper-eaai/paper.tex:152-156` | 없음 |
| 121–422 Hz is the causal leak band | Supported | `results/robust_bands_results.json`, `results/causal_bands_results.json`, `paper-eaai/paper.tex:158-166` | attribution convergence는 보이나 causal language는 여전히 observational 한계 존재 |
| Tree-only ensemble reaches 0.4326 macro-F1 | Supported | `results/ensemble_results.json:24-31`, `paper-eaai/paper.tex:978-1009` | 없음 |
| Strongest deep alternative is MLP GroupDRO 0.358 ± 0.013 over 5 seeds | Supported | `results/multiseed_summary.json`, `paper-eaai/paper.tex:831-839,904-915` | baseline variance coverage가 deep models 위주 |
| County-level LOCO slightly favors causal features when test-size weighted | Supported | `results/sigungu_ood_results.json`, `paper-eaai/paper.tex:688-700` | 16/36 eligible folds만 해당 |
| LightGBM’s single-seed lead shrinks to within noise under multi-seed comparison | Unsupported | `paper-eaai/paper.tex:932-935` vs single-seed-only `src/run_extra_baselines.py:33-35` | multi-seed evidence 부재 |
| Deployment recipe achieves 100% detection at 2.22% FAR | Weak / Contradicted by protocol | `results/cost_metrics_results.json`, `src/run_cost_metrics.py:92-105`, `paper-eaai/paper.tex:1112-1117` | FAR threshold tuned on test, target 1% FAR 미달 |
| Deployment candidate lies on Pareto frontier | Unsupported | `paper-eaai/paper.tex:1096-1101` vs synthetic point in `src/make_fig14_pareto.py:63-76` | direct benchmark 부재 |

## 11. Missing information / blocked checks

- full clean-room rerun은 수행하지 못했다. 이유: 사용자 지시대로 고비용 전체 학습은 피했고, 현재 감사는 existing artifact + smoke test 중심으로 수행했다.
- raw AI Hub 원천 메타데이터/수집 이벤트 정보가 저장소 안에 없어서 duplicate physical event leakage, subject/event-level contamination은 prepared CSV 이상으로는 검증할 수 없었다. 막힌 이유는 `src/data.py`가 바로 가공된 CSV 두 개만 읽기 때문이다.
- Docker clean build와 `paper-eaai` PDF의 end-to-end 재생성은 실행하지 않았다. 막힌 이유는 고비용/시간과, 현재 공식 스크립트가 `paper-eaai`를 겨냥하지 않기 때문이다.
- 운영 threshold의 calibration split 정책은 존재하지 않는다. 그래서 “test-free deployment threshold” 검증은 구조적으로 불가능했다.
- tree baseline multi-seed variance는 artifact가 없어 검증할 수 없었다. 필요한 것은 LightGBM/CatBoost/XGBoost seed sweep 결과다.

## 12. Prioritized repair plan

### Must fix before submission

1. EAAI 제출 경로를 단일 canonical pipeline으로 통합  
   - expected impact: 매우 큼  
   - estimated effort: 중간  
   - confidence that it resolves the issue: 높음

2. E5 운영 metric을 validation-only threshold selection으로 재설계하고 논문 문장 전면 교정  
   - expected impact: 매우 큼  
   - estimated effort: 중간  
   - confidence that it resolves the issue: 높음

3. Fig. 14 / cost benchmark를 실측값 기반으로 다시 생성하고 synthetic proxy 제거  
   - expected impact: 매우 큼  
   - estimated effort: 중간  
   - confidence that it resolves the issue: 높음

4. LightGBM “within noise” 문구 삭제 또는 multi-seed baseline 추가  
   - expected impact: 큼  
   - estimated effort: 낮음(삭제) ~ 중간(실험 추가)  
   - confidence that it resolves the issue: 높음

5. E3/E4 비교의 budget/hyperparameter fairness 정렬  
   - expected impact: 큼  
   - estimated effort: 중간  
   - confidence that it resolves the issue: 중간

### Should fix before release

1. `run_multiseed.sh`를 hard-fail / 5-seed complete-run / portable interpreter로 수정  
   - expected impact: 큼  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 높음

2. `scripts/verify.sh`와 `Makefile`을 `paper-eaai` 기준으로 교체  
   - expected impact: 큼  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 높음

3. `paper-eaai/claims.csv`와 summary docs 최신화  
   - expected impact: 중간  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 높음

4. data root를 환경변수/CLI로 외부화  
   - expected impact: 중간  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 높음

### Nice-to-have

1. tree baselines multi-seed variance 추가  
   - expected impact: 중간  
   - estimated effort: 중간  
   - confidence that it resolves the issue: 중간

2. deterministic policy 문서화 및 GPU determinism 옵션 추가  
   - expected impact: 중간  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 중간

3. stale audit docs 정리  
   - expected impact: 낮음  
   - estimated effort: 낮음  
   - confidence that it resolves the issue: 높음

## 13. Likely reviewer attacks

1. “Your operational FAR result is invalid because you tune threshold on the test fold itself. Why should I trust any deployment claim built on that?”
2. “The repository’s official reproduction path builds a different paper than the one being submitted. Which manuscript is the code actually supporting?”
3. “You say LightGBM’s edge shrinks within noise, but you only ran a single seed. That statement is unsupported.”
4. “Your Pareto frontier includes a synthetic ensemble point with approximated latency/RAM rather than a measured benchmark. This is not acceptable evidence.”
5. “Domain-alignment baselines are not cleanly controlled if training budget/early stopping differ from the claimed comparison backbone.”

## 14. Bottom-line decision

- Technically sound: No
- Reproducible: No
- Leakage-free evaluation: No
- Ablations trustworthy: Partial
- Metrics trustworthy: Partial
- Claim-to-code alignment: No
- Submission-ready for a serious AI/ML venue: Not yet
