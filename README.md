# Acoustic sarcasm detection across languages

Code for the paper *How Well Do Acoustic Sarcasm Models Cross Language Barriers? Evaluating Acoustic Representations and Joint Multilingual Learning*.

The scripts extract prosodic descriptors, openSMILE feature sets and self-supervised speech embeddings from four sarcasm corpora (Dutch, English, German and Mandarin Chinese), and train and evaluate the monolingual, zero-shot and joint multilingual models reported in the paper.

## Data

The corpora are not included.

| Corpus | Language | Source | Utterances | Speakers or units |
|---|---|---|---|---|
| Jansen and Chen (2020) | Dutch | not publicly available | 1,771 | 20 speakers |
| MUStARD++ (Ray et al., 2022) | English | from its authors | 1,201 | 27 speakers |
| MCSD v1 (Gao et al., 2025) | Mandarin | DataverseNL, doi:10.34894/A0NLTQ | 2,625 of 2,705 clips, 2,535 with all seven prosodic features | 135 speakers |
| MuSaG (Scott et al., 2025) | German | metadata from its authors, audio from YouTube | 214 | 22 source videos |

The counts are those after feature extraction in the paper; a corpus prepared later may differ where source videos have changed. Place the corpora as follows:

```
data/corpora/dutch/dataset.csv
data/corpora/dutch/utterance_audio/*.wav
data/corpora/mustard_plus_plus/mustard++_text.csv
data/corpora/mustard_plus_plus/final_utterance_videos/*.mp4
data/corpora/mcsd/Files/MCSD_v1.xlsx
data/corpora/mcsd/clips/*.wav
data/corpora/musag/dataset.csv
data/corpora/musag/dataset.json
data/corpora/musag/segments/*.wav
```

All paths are set in `config.py`. Extracted features are written to `data/features/`, result tables to `results/tables/` and figures to `results/figures/`. `results/tables/` already holds the tables behind every number in the paper, computed on all four corpora; rerunning a script overwrites its table.

### Preparing the German and Mandarin audio

MuSaG and MCSD give YouTube sources and timestamps rather than audio, so the clips are cut from the source videos. `ffmpeg` and `yt-dlp` must be on the path.

```
python prepare/musag.py
python prepare/mcsd.py
```

`prepare/musag.py` downloads the 22 videos listed in `dataset.csv` and cuts the 214 segments as 16 kHz mono WAV. `prepare/mcsd.py` downloads the episodes listed in `prepare/mcsd_episodes.csv`, which gives the YouTube video of each MCSD item we could obtain, and cuts the clips listed in `MCSD_v1.xlsx`. Of the 80 clips we could not cut, 24 belong to `tkxdh_s4_ep7_1` and have no start or end time in `MCSD_v1.xlsx`, and 56 belong to the 31 items whose episodes we could not obtain: `tkxdh_s4_ep10_3` to `tkxdh_s4_ep10_30`, `tkxdh_s5_ep5_1`, `tkxdh_s5_ep5_2` and `tkxdh_s5_ep7_2`.

### Without the Dutch corpus

Every extraction script takes `--corpus` and skips a corpus whose files are missing. The cross-lingual experiments, the statistics and the scripts named `dutch_*` need all four corpora. With the three public corpora, `experiments/baselines.py`, `separability.py`, `monolingual.py` and `budget_curves.py` run on English, Mandarin and German.

## Setup

Python 3.12 with the packages in `requirements.txt`, and `ffmpeg` on the path (the English audio is read from video files). Training runs on a CUDA GPU if one is available. The pre-trained encoders are downloaded from the Hugging Face Hub on first use.

```
pip install -r requirements.txt
```

## Running

Run the scripts from the repository root, in this order. Later steps read the tables written by earlier ones.

```
python extraction/prosody.py
python extraction/mfcc.py
python extraction/dutch_feature_tables.py
python extraction/dutch_mfcc128.py
python extraction/embeddings.py

python experiments/baselines.py
python experiments/effect_sizes.py
python experiments/separability.py
python experiments/corpus_checks.py corpus
python experiments/monolingual.py
python experiments/budget_curves.py
python experiments/xlsr_ceiling.py
python experiments/dutch_benchmarks.py
python experiments/leakage.py
python experiments/zeroshot.py
python experiments/zeroshot_standardised.py
python experiments/zeroshot_seed_mean.py
python experiments/language_tax.py
python experiments/joint.py
python experiments/joint_vs_specialists.py
python experiments/lolo.py
python experiments/controls.py pooled
python experiments/controls.py budget
python experiments/corpus_checks.py perlang
python experiments/contrastive_ablation.py
python experiments/size_matched.py
python experiments/selection_sensitivity.py
python experiments/svm_grid.py
python experiments/statistics.py
python experiments/feature_importance.py
python experiments/layerwise.py extract
python experiments/layerwise.py extract-e2v
python experiments/layerwise.py sweep
python experiments/layerwise.py report
python experiments/layerwise.py pipeline
python experiments/layerwise.py compare
python experiments/layerwise.py transfer
python experiments/layerwise.py joint
python experiments/vocal_separation.py separate
python experiments/vocal_separation.py extract
python experiments/vocal_separation.py bench

python figures/pipeline.py
python figures/effect_heatmap.py
python figures/results.py
```

Each score list in `monolingual.csv` and `dutch_benchmarks.csv` comes with the draws it belongs to, and paired tests join two lists on their common draws.

## Results and the scripts that produce them

| Paper | Script | Output in `results/tables/` |
|---|---|---|
| Table 1, majority-class F1; Section 5.2, random baseline | `experiments/baselines.py` | `chance_baselines.csv` |
| Table 2, Silhouette scores | `experiments/separability.py` | `silhouette.csv` |
| Table 3 and Figure 4, monolingual results | `experiments/monolingual.py`, `experiments/dutch_benchmarks.py` | `monolingual.csv`, `dutch_benchmarks.csv` |
| Table 4, zero-shot transfer | `experiments/zeroshot.py`, `zeroshot_standardised.py`, `zeroshot_seed_mean.py` | `zeroshot_prosodic.csv`, `zeroshot_standardised.csv`, `zeroshot_seed_mean.csv` |
| Section 5.4, language tax | `experiments/language_tax.py` | `language_tax.csv` |
| Table 5, joint multilingual training | `experiments/joint.py`, `controls.py pooled`, `corpus_checks.py perlang` | `joint.csv`, `pooled.csv`, `per_language.csv` |
| Figure 5, joint model against per-language models | `experiments/joint_vs_specialists.py` | `joint_vs_specialists.csv` |
| Section 5.5, leave-one-language-out | `experiments/lolo.py` | `lolo.csv` |
| Section 5.6, the contrastive stage | `experiments/contrastive_ablation.py`, `dutch_benchmarks.py` | `contrastive_ablation.csv`, `dutch_geometry.csv` |
| Table 6, our rows | `experiments/monolingual.py` | `monolingual.csv` |
| Section 5.7, the published classifier grid on our embeddings | `experiments/svm_grid.py` | `svm_grid.csv` |
| Figure 1 | `figures/pipeline.py` | |
| Figure 2 | `experiments/feature_importance.py`, `figures/results.py` | `feature_importance.csv` |
| Figure 3 and Section 5.1, effect sizes | `experiments/effect_sizes.py`, `figures/effect_heatmap.py` | `effect_sizes_core7.csv`, `effect_sizes_all14.csv`, `effect_sizes_with_mfcc.csv` |
| Sections 4.2 and 5.1, speaker identity, cue correlations, Dutch utterance types | `experiments/corpus_checks.py corpus` | `speaker_balance.csv`, `cue_correlations.csv`, `dutch_utterance_types.csv` |
| Section 5.3, encoders of equal size | `experiments/size_matched.py` | `size_matched.csv` |
| Significance tests, equivalence bounds and effect-size intervals | `experiments/statistics.py` | `corrected_tests.csv`, `pooling_equivalence.csv`, `cohen_d_intervals.csv` |
| Section 4.3, training budget in zero-shot transfer | `experiments/controls.py budget` | `budget.csv` |
| Supplementary S1, speaker leakage | `experiments/leakage.py` | `leakage.csv` |
| Supplementary S2, the training budget | `experiments/monolingual.py`, `budget_curves.py`, `xlsr_ceiling.py`, `corpus_checks.py corpus` | `monolingual.csv`, `budget_curves.csv`, `budget_selection.csv`, `xlsr_ceiling.csv`, `held_out_sizes.csv` |
| Supplementary S3, Dutch pairwise comparisons | `experiments/dutch_benchmarks.py`, `statistics.py` | `dutch_pairwise.csv`, `corrected_tests.csv` |
| Supplementary S4, equivalence bounds for the joint model | `experiments/statistics.py` | `pooling_equivalence.csv` |
| Supplementary S5, leave-one-language-out per fold | `experiments/lolo.py` | `lolo.csv` |
| Supplementary S6 and Figure S1, encoder layers | `experiments/layerwise.py` | `layerwise_*.csv` |
| Supplementary S7, feature selection on training speakers | `experiments/selection_sensitivity.py` | `selection_sensitivity.csv` |
| Supplementary S8, effect-size intervals | `experiments/statistics.py` | `cohen_d_intervals.csv` |
| Supplementary S9 and Figure S2, the contrastive stage | `experiments/contrastive_ablation.py`, `dutch_benchmarks.py`, `controls.py pooled` | `contrastive_ablation.csv`, `dutch_geometry.csv`, `pooled.csv` |
| Section 6.4, vocal separation of the English audio | `experiments/vocal_separation.py` | `vocal_separation.csv` |
