# Interpreting Multi-Horizon Time Series Deep Learning Models

Interpreting the model's behavior is important in understanding decision-making in practice. However, explaining complex time series forecasting models faces challenges due to temporal dependencies between subsequent time steps and the varying importance of input features over time. Many time series forecasting models use input context with a look-back window for better prediction performance. However, the existing studies (1) do not consider the temporal dependencies among the feature vectors in the input window and (2) separately consider the time dimension that the feature dimension when calculating the importance scores. In this work, we propose a novel **Windowed Temporal Saliency Rescaling** method to address these issues. 

## Citation

Find our paper on arxiv at https://arxiv.org/pdf/2412.04532. Please cite the following if you use our work.

```
@article{islam2024wintsr,
  title={WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting Time Series Deep Learning Models},
  author={Islam, Md Khairul and Fox, Judy},
  journal={arXiv preprint arXiv:2412.04532},
  year={2024}
}
```

## Core Libraries

The following libraries are used as a core in this framework.

### [Captum](https://captum.ai/docs/introduction)

(“comprehension” in Latin) is an open source library for model interpretability built on PyTorch.

### [Time Interpret (tint)](https://josephenguehard.github.io/time_interpret/build/html/index.html)

Expands the Captum library with a specific focus on time-series. It includes various interpretability methods specifically designed to handle time series data.

### [Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)

TSlib is an open-source library for deep learning researchers, especially deep time series analysis.

## Interpretation Methods

The following local intepretation methods are supported at present:
<details open>

1. *WinTSR* - proposed new method
2. *Feature Ablation* [[2017]](https://arxiv.org/abs/1705.08498)
3. *Dyna Mask* [[ICML 2021]](https://arxiv.org/abs/2106.05303)
4. *Extremal Mask* [[ICML 2023]](https://proceedings.mlr.press/v202/enguehard23a/enguehard23a.pdf)
5. *Feature Permutation* [[Molnar 2020]](https://christophm.github.io/interpretable-ml-book/)
6. *Augmented Feature Occlusion* [[NeurIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/08fa43588c2571ade19bc0fa5936e028-Paper.pdf)
7. *Gradient Shap* [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
8. *Integreated Gradients* [[ICML 2017]](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
9. *WinIT* [[ICLR 2023]](https://openreview.net/forum?id=C0q9oBc3n4)
10. *TSR* [[NeurIPS 2020]](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf)
11. *ContraLSP* [[ICLR 2024]](https://arxiv.org/pdf/2401.08552)

</details>

## Foundation Models

We support the following time series LLM models

1. GPT4TS - [One Fits All (OFA) : Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939) (NeurIPS 2023 Spotlight)
2. CALF - [CALF - Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning.](https://arxiv.org/abs/2403.07300) (Under review 2024)
3. TimeLLM - [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/pdf/2310.01728) (ICLR 2024)

## Time Series Models

This repository currently supports the following models:

<details open>

- [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
- [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
- [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
- [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
- [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
- [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
- [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
- [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
- [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py)
- [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)
- [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
- [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
- [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
- [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
- [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py)
- [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py)
- [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
- [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)
- [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
- [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)
- [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
- [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
- [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
- [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)
  
</details>

## Train & Test

Use the [run.py](/run.py) script to train and test the time series models. Check the [scripts](/scripts/) and [slurm](/slurm/) folder to see sample scripts. Make sure you have the datasets downloaded in the `dataset` folder following the `Datasets` section. Following is a sample code to train the electricity dataset using the DLinear model. To test an already trained model, just remove the `--train` parameter.

```bash
python run.py \
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model DLinear \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1
```

## Interpret

Use the [interpret.py](/interpret.py) script to interpret a trained model. Check the [scripts](/scripts/) and [slurm](/slurm/) folder to see more sample scripts. Following is a sample code to interpret the `iTransformer` model trained on the electricity dataset using using some of the interpretation methods. This evaluates the 1st iteration among the default 3 in the result folder.

```bash
python interpret.py \
  --task_name long_term_forecast \
  --explainers feature_ablation augmented_occlusion feature_permutation integrated_gradients gradient_shap wtsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model iTransformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --itr_no 1
```

## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. Only `mimic-iii` dataset is private and hence must be approved to get access from [PhysioNet](https://mimic.mit.edu/docs/gettingstarted/).

### Electricity

The electricity dataset [^1] was collected in 15-minute intervals from 2011 to 2014. We select the records from 2012 to 2014 since many
zero values exist in 2011. The processed dataset contains
the hourly electricity consumption of 321 clients. We use
’MT 321’ as the target, and the train/val/test is 12/2/2 months. We aggregated it to 1h intervals following prior works.  

### Traffic

This dataset [^2] records the road occupancy rates from different sensors on San Francisco freeways.

### Mimic-III

MIMIC-III is a multivariate clinical time series dataset with a range of vital and lab measurements taken over time for around 40,000 patients at the Beth Israel Deaconess Medical Center in Boston, MA (Johnson et al. [^3], 2016). It is widely used in healthcare and medical AI-related research. There are multiple tasks associated, including mortality, length-of-stay prediction, and phenotyping. 

<details>

We follow the pre-processing procedure described in Tonekaboni et al. (2020) [^4] and use 8 vitals and 20 lab measurements hourly over a 48-hour period to predict patient mortality. For more visit the [source description](https://physionet.org/content/mimiciii/1.4/).

This is a private dataset. Refer to [the official MIMIC-III documentation](https://mimic.mit.edu/iii/gettingstarted/dbsetup/). ReadMe and datagen of MIMIC is from [Dynamask Repo](https://github.com/JonathanCrabbe/Dynamask). This repository followed the database setup instructions from [the offficial site here](https://mimic.mit.edu/docs/gettingstarted/local/install-mimic-locally-windows/). 

- Run this command to acquire the data and store it:
  
   ```shell
   python -m data.mimic_iii.icu_mortality --sqluser YOUR_USER --sqlpass YOUR_PASSWORD
   ```

  If everything happens properly, two files named ``adult_icu_vital.gz`` and `adult_icu_lab.gz`
  are stored in `dataset/mimic_iii`.

- Run this command to preprocess the data:
   ```shell
   python -m data.mimic_iii.data_preprocess
   ```
  If everything happens properly, a file `mimic_iii.pkl` is stored in `dataset/mimic_iii`.
 
<!-- > Before data was incorporated into the MIMIC-III database, it was first deidentified in accordance with Health Insurance Portability and Accountability Act (HIPAA) standards using structured data cleansing and date shifting. The deidentification process for structured data required the removal of all eighteen of the identifying data elements listed in HIPAA, including fields such as patient name, telephone number, address, and dates. In particular, dates were shifted into the future by a random offset for each individual patient in a consistent manner to preserve intervals, resulting in stays which occur sometime between the years 2100 and 2200. Time of day, day of the week, and approximate seasonality were conserved during date shifting. Dates of birth for patients aged over 89 were shifted to obscure their true age and comply with HIPAA regulations: these patients appear in the database with ages of over 300 years. -->
</details>

## Reproduce

The module was developed using python 3.10.

### Option 1. Use Container

[Dockerfile](/Dockerfile) contains the docker buidling definitions. You can build the container using 
```
docker build -t timeseries
```
This creates a docker container with name tag timeseries. The run our scripts inside the container. To create a `Singularity` container use the following [definition file](/singularity.def).
```
sudo singularity build timeseries.sif singularity.def
```
This will create a singularity container with name `timeseries.sif`. Note that, this requires `sudo` privilege.

### Option 2. Use Virtual Environment

First create a virtual environment with the required libraries. For example, to create an venv named `ml`, you can either use the `Anaconda` library or your locally installed `python`. An example code using Anaconda,

```bash
conda create -n ml python=3.10
conda activate ml
```

This will activate the venv `ml`. Install the required libraries,

```bash
python3 -m pip install -r requirements.txt
```

If you want to run code on your GPU, you need to have CUDA installed. Check if you already have CUDA installed. 

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} backend')
```

If this fails to detect your GPU, install CUDA using,

```bash
pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu118
```

## References
<!-- https://docs.github.com/en/enterprise-cloud@latest/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#footnotes -->
[^1]: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.

[^2]: https://pems.dot.ca.gov/.

[^3]: Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 2016.

[^4]: Sana Tonekaboni, Shalmali Joshi, Kieran Campbell, David K Duvenaud, and Anna Goldenberg. What went wrong and when? Instance-wise feature importance for time-series black-box models. In Neural Information Processing Systems, 2020.