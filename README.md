# Breast Tumor Classification on BreakHis

CSCI 6379 Intro to Deep Learning, Fall 2026 · Group 13

Oscar Juarez · Adrian Ramos · Gabriel Herrera-Arteaga

## Overview

Deep learning models are often reported to classify breast tumor biopsy images with over 90% accuracy. We test how well that holds up under realistic conditions on the BreakHis dataset. We compare a small CNN with a large one, handle the class imbalance, and evaluate only on patients the model has never seen. We then try to close the gap for the small model with knowledge distillation and stain augmentation, and compare our results with published BreakHis work.

## Research questions

| | Question | What we compare | Metrics |
|:--|:--|:--|:--|
| RQ1 | Does a small CNN match a large one? | MobileNetV2 vs ResNet-50 on the same patient split | Balanced accuracy, AUROC, CPU time per image |
| RQ2 | Does handling class imbalance help catch benign tumors? | Plain loss vs class-weighted loss vs oversampling | Benign recall, macro F1 |
| RQ3 | How much does a random image split inflate results? | Image-level split vs patient-level split | Gap in accuracy and macro F1 |
| RQ4 | Can we make the small model better? | MobileNetV2 with stain augmentation and distillation from ResNet-50 | All of the above, plus published results |

## Data

BreakHis v1 has 7,909 microscope images from 82 patients at 40X, 100X, 200X and 400X. Each image is labeled benign or malignant. The patient ID is read from each file name, so the default split keeps every patient in exactly one of train, validation or test, at roughly 70, 15 and 15 percent of patients.

The images are not stored in this repo. Download them from Kaggle in Colab or locally:

```python
import kagglehub
DATA_DIR = kagglehub.dataset_download("kurshidbasheer/breakhisbreast-cancer-histopathological-images")
```

If the download includes the official `Folds.csv`, add `--fold 1` through `--fold 5` to train on the original authors' patient-level folds. That makes our numbers directly comparable with published results.

## Getting started

The easiest way to run everything is the Colab notebook in `notebooks/colab_quickstart.ipynb`. Open it in Colab, switch the runtime to GPU and run the cells in order. It clones this repo, downloads the data, checks the split and runs a short sanity test.

To run locally instead:

```bash
pip install -r requirements.txt
python train.py --data-dir /path/to/BreaKHis --model mobilenet_v2 --run-name e1_mobilenet_s0
```

Every run saves its settings and test metrics to `results/<run-name>.json` and its best model to `checkpoints/<run-name>.pt`. Checkpoints are too large for git and stay local.

## Experiments

| Run | Question | Command flags |
|:--|:--|:--|
| e1_mobilenet | RQ1 | `--model mobilenet_v2` |
| e2_resnet50 | RQ1 | `--model resnet50` |
| e3_weighted | RQ2 | `--model mobilenet_v2 --loss weighted` |
| e4_oversample | RQ2 | `--model mobilenet_v2 --sampler oversample` |
| e5_image_split | RQ3 | `--model mobilenet_v2 --split image` |
| e6_stain | RQ4 | `--model mobilenet_v2 --stain-aug` |
| e7_distill | RQ4 | `--model mobilenet_v2 --teacher checkpoints/e2_resnet50_s0.pt` |
| e8_distill_stain | RQ4 | `--model mobilenet_v2 --teacher checkpoints/e2_resnet50_s0.pt --stain-aug` |

Run all of them for three seeds with:

```bash
bash scripts/run_experiments.sh /path/to/BreaKHis 15
```

Then build the results table:

```bash
python collect_results.py
```

This writes `results/summary.md` and `results/summary.csv`, with the mean and standard deviation across seeds.

## Repo layout

```
breakhis/
  data.py        file indexing, patient-level splits, stain augmentation, dataset
  models.py      MobileNetV2, ResNet-50 and EfficientNet-B0 with a 2-class head
  metrics.py     image-level and patient-level metrics
train.py         trains and evaluates one experiment
collect_results.py   builds the results table
scripts/run_experiments.sh   runs every experiment
notebooks/colab_quickstart.ipynb   Colab setup and first run
results/         one JSON file per run, committed to the repo
```

## Results

Coming soon.

## Comparison with prior work

| Work | Method | Split | Reported result |
|:--|:--|:--|:--|
| Spanhol et al., 2016 | Hand-crafted texture features | Patient-level | 80 to 85% accuracy |
| Han et al., 2017 | Class structure-based deep CNN | Eight-subtype task | 93.2% average accuracy |
| Ours | MobileNetV2, distilled and stain-augmented | Patient-level | Coming soon |

## References

- Fahim et al., 2025. Artificial intelligence in healthcare and medicine: clinical applications, therapeutic advances, and future perspectives. European Journal of Medical Research.
- Spanhol et al., 2016. A Dataset for Breast Cancer Histopathological Image Classification. IEEE Transactions on Biomedical Engineering.
- Han et al., 2017. Breast Cancer Multi-classification from Histopathological Images with Structured Deep Learning Model. Scientific Reports.
- Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
- He and Garcia, 2009. Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering.
- Hinton et al., 2015. Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Tellez et al., 2018. Whole-slide mitosis detection in H&E breast histology using PHH3 as a reference to train distilled stain-invariant convolutional networks. IEEE Transactions on Medical Imaging.
