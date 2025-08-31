# Anesthesia Induction Dosage Modeling

A comprehensive machine learning prototype for predicting anesthesia induction parameters using multi-stage modeling approach. This project predicts anesthesia type, dosage, and early response based on patient demographics, genetic factors, medical history, and physiological parameters.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanations](#model-explanations)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Overview

This project implements a sophisticated anesthesia induction system that uses separate machine learning models for different stages of anesthesia planning:

1. **Anesthesia Type Selection** - Predicts optimal anesthetic agent (Propofol, Sevoflurane, Isoflurane, Desflurane, Ketamine)
2. **Dosage Calculation** - Predicts precise dosage based on patient profile
3. **Response Prediction** - Predicts early response (Effective, Ineffective, Adverse)

The system incorporates pharmacogenomic data, patient physiology, and medical history to provide personalized anesthesia recommendations with explainable AI capabilities.

## ✨ Features

- **Multi-Stage Prediction**: Separate specialized models for type, dosage, and response
- **Pharmacogenomic Integration**: Incorporates CYP450 enzymes, genetic variants, and drug metabolism pathways
- **Explainable AI**: SHAP-based explanations for all predictions
- **Comprehensive Feature Set**: 25+ patient parameters including demographics, genetics, and medical history
- **High Accuracy**: CatBoost models with optimized hyperparameters
- **Synthetic Dataset**: 40,000+ synthetic patient records for robust training

## 🏗️ Architecture

```
├── predictor.ipynb          # Main prediction engine with SHAP explanations
├── Anesthesia_type_model.ipynb     # Type selection model training
├── Anesthesia_dosages_model.ipynb  # Dosage prediction model training
├── Anesthesia_response_model.ipynb # Response prediction model training
├── feature_importance_map.py       # Feature explanation templates
├── saved_model/                    # Trained model artifacts
│   ├── anesthesia_type_model_catBoost_upsampled.cbm
│   ├── general_dosage_model.cbm
│   ├── general_response_confidence_model.cbm
│   └── *.pkl (metadata files)
├── catboost_info/                  # Training logs and metrics
└── anesthesia_dataset_v4.csv       # Training dataset
```

## 📊 Dataset

> **⚠️ Important Disclaimer**: This dataset contains **synthetic data only** and is not based on real patient records. The data was generated specifically to prove the hypothesis that machine learning models can effectively predict anesthesia induction dosages by integrating patient demographics, pharmacogenomic (PGx) profiles, and clinical parameters. This is a research prototype for demonstrating ML feasibility in anesthesia planning and should not be used for clinical decision-making.

The synthetic dataset includes the following features:

### Patient Demographics

- Age, Gender, Height, Weight, BMI, IBW, ABW
- Diet preferences

### Medical History

- Organ Function, Kidney Function, Cardiovascular History
- Diabetes, Current Medications, Procedure Type, ASA Class

### Pharmacogenomic Factors

- ALDH2 Genotype, CYP2D6/CYP3A4/CYP2C9/CYP2B6 Types
- UGT1A1, RYR1, SCN9A, F5, GABRA2, OPRM1 Variants

### Target Variables

- **General_AnesthesiaType**: Propofol, Sevoflurane, Isoflurane, Desflurane, Ketamine
- **General_Dosage**: Numerical dosage in mg
- **General_Response**: Effective, Ineffective, Adverse

## 🤖 Models

### 1. Anesthesia Type Model

- **Algorithm**: CatBoost Classifier (Multi-class)
- **Features**: 24 patient parameters
- **Classes**: 5 anesthesia types
- **Technique**: Upsampling for class balance

### 2. Dosage Model

- **Algorithm**: CatBoost Regressor
- **Features**: 25 parameters (including predicted type)
- **Target**: Continuous dosage prediction
- **Metric**: RMSE optimization

### 3. Response Model

- **Algorithm**: CatBoost Classifier (Multi-class)
- **Features**: 26 parameters (including type and dosage)
- **Classes**: Effective, Ineffective, Adverse
- **Technique**: Custom thresholds for confidence

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
pip
```

### Dependencies

```bash
pip install catboost scikit-learn pandas numpy shap matplotlib seaborn joblib
```

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Faerque/anesthesia-project.git
cd anesthesia-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Prediction

```python
from predictor import predict_all
import pandas as pd

# Load patient data
patient_data = pd.DataFrame({
    'Age': [35],
    'Gender': ['Female'],
    'Height_cm': [165],
    'Weight_kg': [65],
    # ... other features
})

# Make prediction
result = predict_all(patient_data, explain=True)
print(result)
```

### Output Format

```json
{
  "General_AnesthesiaType": "Propofol",
  "General_Dosage": 125.3,
  "General_Response": "Effective",
  "Patient_Profile": {...},
  "SHAP_Explanations": {
    "General_AnesthesiaType": {
      "explanations": ["Age (35) favored Propofol due to better hemodynamic stability..."],
      "per_class_shap_values": {...}
    },
    "General_Dosage": {...},
    "General_Response": {...}
  },
  "SHAP_Influence_By_Feature": {...}
}
```

## 🔍 Model Explanations

The system provides detailed explanations using SHAP (SHapley Additive exPlanations):

### Feature Impact Examples

**Age**: Younger patients (<40) often receive higher doses due to efficient drug clearance
**BMI**: Normal BMI (18.5-25) allows standard dosing; extremes require adjustment
**CYP2D6**: Ultra-metabolizers may need higher doses; poor metabolizers need lower doses
**ASA Class**: Higher classes (3-4) indicate reduced dosing due to comorbidities

### Pharmacogenomic Considerations

- **CYP2D6 PM**: Reduced metabolism → lower dosage
- **CYP3A4 UM**: Enhanced metabolism → higher dosage
- **RYR1 Variant**: Risk of malignant hyperthermia → prefer Propofol
- **ALDH2 Deficient**: Slower aldehyde metabolism → careful dosing

## 📈 Performance

### Anesthesia Type Model

- **Accuracy**: 87.3%
- **F1-Score**: 0.85 (weighted)
- **Classes**: Balanced performance across 5 types

### Dosage Model

- **RMSE**: 6.77 mg
- **MAE**: 4.92 mg
- **R²**: 0.823

### Response Model

- **Accuracy**: 82.1%
- **Precision (Adverse)**: 0.89
- **Recall (Effective)**: 0.91

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update tests for new features
- Ensure models maintain or improve performance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

### Pharmacogenomic References

- Zhou SF, et al. "Clin Pharmacokinet." 2009;48(11):761–804
- Wilkinson GR. "Clin Pharmacokinet." 2005;44(4):279–295
- Lee CR, et al. "Pharmacogenomics." 2002;3(2):277–287

### Clinical References

- Aldrete JA. "Anesth Analg." 1998;86(4):791–795
- Ingrande J, Lemmens HJ. "Br J Anaesth." 2010;105(Suppl 1):i16–i23
- Daabiss M. "Saudi J Anaesth." 2011;5(2):112–116

### Machine Learning References

- CatBoost: Prokhorenkova L, et al. "Analysis of CatBoost." 2018
- SHAP: Lundberg SM, Lee SI. "A Unified Approach to Interpreting Model Predictions." 2017

## 🙏 Acknowledgments

- Synthetic dataset generated for research purposes
- Pharmacogenomic data based on clinical literature
- CatBoost framework for robust gradient boosting
- SHAP library for model interpretability

## 📞 Contact

**Faerque**

- GitHub: [@Faerque](https://github.com/Faerque)
- Project: [Anesthesia Induction Modeling](https://github.com/Faerque/anesthesia-project)

---

_This is a research prototype. Not intended for clinical use without proper validation and regulatory approval._
