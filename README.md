# Age-Specific Bias Mitigation in OSA Detection (SIT723/SIT729, Deakin University)

This repository implements and compares two representation-level debiasing
methods — a **Domain-Adversarial Neural Network (DANN)** and **Fair Supervised
Contrastive Learning (FSCL)** — for mitigating age-specific bias in deep
learning-based obstructive sleep apnea (OSA) detection from single-lead ECG.
Both methods are built on top of a shared CNN–Transformer–LSTM baseline
(Pham et al. 2025) and evaluated on the PhysioNet Apnea-ECG database.

## Project Details

| **Field** | **Details** |
| :--- | :--- |
| **Institution** | Deakin University |
| **Student** | [Trung Hoang Anh (Andre) Nguyen](https://www.linkedin.com/in/andre-nguyen-0298a9287/) |
| **Supervisor** | [Dr. Md. Ahsan Habib](https://experts.deakin.edu.au/50940-ahsan-habib) |

---

## Core Methodology

The study addresses age-specific bias in ECG-based OSA detection, where
diagnostic performance (sensitivity and specificity) can differ substantially
between younger and older cohorts. All models share the same
**CNN–Transformer–LSTM** backbone; the debiasing methods differ in how they
shape the learned representation:

* **Baseline** — the CNN–Transformer–LSTM classifier without any fairness
  mechanism.
* **DANN** — adds a **Gradient Reversal Layer (GRL)** and an age discriminator
  that reads from a shared bottleneck, so the feature extractor is trained to
  produce age-invariant representations while preserving apnea-discriminative
  information. A class-conditional adversary is used to target both the
  sensitivity and specificity gaps between age groups.
* **FSCL** — Fair Supervised Contrastive Learning: pulls together same-class
  samples across age groups and pushes apart different-class samples,
  encouraging the embedding to be organised by apnea status rather than by age.

The two demographic cohorts are:

* **Young:** $< 40$ years
* **Old:** $\geq 40$ years

---

## Repository Structure

* `preprocessing.py`
    * Standardises the raw PhysioNet Apnea-ECG signals, extracts one-minute
      segments with margin context, performs R-peak detection via the
      **Hamilton algorithm** with correction, and derives the R-peak amplitude
      and R–R interval (RRI) channels via cubic-spline interpolation.
    * Attaches age labels to every segment for the debiasing methods.

* `baseline.py`
    * The CNN–Transformer–LSTM baseline classifier (Pham et al. 2025), used as
      the reference and as the shared backbone for DANN and FSCL.

* `DANN.py`
    * The dual-head domain-adversarial model: an apnea classifier and an age
      discriminator connected through a Gradient Reversal Layer, with a
      class-conditional adversary variant for targeting the specificity gap.

* `FSCL.py`
    * Fair Supervised Contrastive Learning built on the same backbone; combines
      a supervised contrastive fairness loss with the apnea classification
      objective.

* `/apnea-ecg-database-1.0.0`
    * Placeholder directory for the [**PhysioNet Apnea-ECG Database**](https://physionet.org/content/apnea-ecg/1.0.0/).
    * **Note:** users must download the data from PhysioNet into this folder
      before running any script.

* `/dataset`
    * Output directory for the preprocessed data (pickle files) produced by
      `preprocessing.py`, and the source that the training scripts read from.

---

## Getting Started

1. Download the PhysioNet Apnea-ECG database into `apnea-ecg-database-1.0.0/`.
2. Run `python preprocessing.py` to generate the preprocessed dataset in
   `dataset/`.
3. Train and evaluate the models:
```bash
   python baseline.py
   python DANN.py
   python FSCL.py
```

---

## References

1. **Baseline model:** Pham, D. T., & Mouček, R. (2025). Efficient sleep apnea
   detection using single-lead ECG with CNN–Transformer–LSTM. *Computers in
   Biology and Medicine*.
2. **DANN:** Ganin, Y., et al. (2016). Domain-adversarial training of neural
   networks. *Journal of Machine Learning Research*.
3. **FSCL:** Park, S., Lee, J., Lee, P., Hwang, S., Kim, D., & Byun, H. (2022).
   Fair Contrastive Learning for Facial Attribute Classification. *CVPR*.
4. **Bottleneck framework:** Chen, X., et al. (2023). BAFNet: Bottleneck
   attention based fusion network for sleep apnea detection. *IEEE JBHI*.
5. **PhysioNet Apnea-ECG dataset:** Goldberger, A., Amaral, L., Glass, L.,
   Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
   PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research
   resource for complex physiologic signals. *Circulation* [Online]. 101 (23),
   pp. e215–e220. RRID:SCR_007345.