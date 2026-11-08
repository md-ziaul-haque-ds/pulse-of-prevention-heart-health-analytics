# ❤️ Pulse of Prevention
### Analyzing Heart Health for Better Outcomes

An end-to-end **Heart Disease Exploratory Data Analysis (EDA) and Machine Learning Project** using Python, statistical analysis and Logistic Regression, featuring an interactive Streamlit application for educational heart disease prediction.

---

## 📌 Project Overview

**Pulse of Prevention** explores heart disease-related patterns in patient health data to understand the relationships between clinical indicators and the target variable.

The project follows a complete data analytics and machine learning workflow, starting with data cleaning and exploratory analysis, followed by statistical hypothesis testing, feature analysis, predictive modelling and business recommendations.

An interactive Streamlit application was also developed to demonstrate how a trained machine learning model can generate predictions using user-provided input values.

**Project Objective:** To explore health-related patterns, identify statistically significant associations and demonstrate the potential application of machine learning in healthcare analytics.

---

## 🎯 Project Objectives

- Understand the structure and characteristics of the heart disease dataset.
- Perform data cleaning and preprocessing.
- Explore relationships between clinical indicators and the target variable.
- Identify important patterns through exploratory data analysis.
- Apply statistical hypothesis testing to evaluate associations.
- Develop and evaluate a Logistic Regression classification model.
- Build an interactive Streamlit prediction application.
- Translate analytical findings into meaningful business insights and recommendations.

---

## 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | SciPy, Statistical Hypothesis Testing |
| Machine Learning | Scikit-learn |
| Predictive Modelling | Logistic Regression |
| Data Application | Streamlit |
| Development Environment | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
Pulse-of-Prevention/
│
├── Streamlit_App/
│   ├── app.py
│   ├── heart_disease.csv
│   └── Run_App.bat
│
├── Visualization EDA/
│   └── EDA Visualization Screenshots
│
├── Heart_Disease_EDA.ipynb
│
├── Pulse of Prevention_User_Interface.png
│
├── Pulse_of_Prevention_Business_Insights.docx
│
└── README.md
```

### 📁 Project Files

- **[Streamlit Application](Streamlit_App/)** — Interactive heart disease prediction application.
- **[EDA Visualizations](Visualization%20EDA/)** — Screenshots of the exploratory data analysis and visual findings.
- **[Jupyter Notebook](Heart_Disease_EDA.ipynb)** — Complete Python-based EDA, statistical analysis and machine learning workflow.
- **[Application Interface](Pulse%20of%20Prevention_User_Interface.png)** — Screenshot of the Streamlit application.
- **[Business Insights & Recommendations](Pulse_of_Prevention_Business_Insights.docx)** — Detailed documentation of findings, recommendations and project conclusions.

---

## 🔍 Project Workflow

The project was completed through the following analytical phases:

### Phase 1 — Project Understanding & Data Loading
- Imported the required Python libraries.
- Loaded the heart disease dataset.
- Examined dataset dimensions, columns and data types.
- Performed initial data inspection.

### Phase 2 — Data Cleaning & Preprocessing
- Checked for missing values and duplicate records.
- Identified and removed duplicate rows.
- Created a cleaned dataset for further analysis.
- Examined categorical and numerical features.

### Phase 3 — Exploratory Data Analysis (EDA)
- Conducted univariate analysis to understand individual feature distributions.
- Performed bivariate analysis to examine relationships between variables.
- Explored target distribution and feature patterns.
- Used visualizations to identify meaningful trends.

### Phase 4 — Statistical Analysis
- Applied correlation analysis to examine numerical relationships.
- Used Chi-Square tests to investigate associations between categorical variables and the target.
- Applied statistical significance testing to support analytical interpretations.

### Phase 5 — Feature Analysis & Interpretation
- Examined important clinical indicators.
- Identified features showing stronger associations with the target.
- Interpreted findings while distinguishing statistical association from causation.

### Phase 6 — Machine Learning Model Development
- Prepared features and target variables.
- Encoded categorical features.
- Applied feature scaling through a Scikit-learn Pipeline.
- Developed a Logistic Regression classification model.
- Evaluated the model using a stratified 80:20 train-test split.

### Phase 7 — Business Insights & Recommendations
- Summarized key analytical findings.
- Converted statistical results into practical recommendations.
- Documented project limitations and final conclusions.

### Phase 8 — Interactive Streamlit Application
- Developed an interactive prediction interface.
- Integrated the Logistic Regression model.
- Created input fields for all 13 dataset features.
- Displayed prediction results and probability.
- Added a custom dashboard design with an animated heartbeat and ECG header.

---

## 📊 Dataset Information

The dataset contains clinical measurements and health-related attributes used for heart disease classification.

**Original Dataset:** 1,025 records

**Records after duplicate removal:** 302 records

**Duplicate occurrences removed:** 723

**Input Features:** 13 original features

**Target Variable:** `target`

### Dataset Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Sex category code |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia category code |
| target | Heart disease classification target |

**Note:** The dataset uses numerical category codes for several clinical variables.

---

## 📈 Key Analytical Findings

Based on the completed EDA and statistical analysis, the following findings were observed:

| No. | Finding | Statistical Evidence |
|---|---|---|
| 1 | Age showed a negative association with the target. | Correlation: -0.221 |
| 2 | Chest pain type (`cp`) showed a statistically significant association with the target. | Chi-Square p-value < 0.001 |
| 3 | Maximum heart rate (`thalach`) showed a positive association with the target. | Correlation: +0.420 |
| 4 | Exercise-induced angina (`exang`) showed a statistically significant association with the target. | Chi-Square p-value < 0.001 |
| 5 | ST depression (`oldpeak`) showed a negative association with the target. | Correlation: -0.429 |
| 6 | Cholesterol did not show a statistically significant association in the test. | p-value: 0.158 |
| 7 | Fasting blood sugar (`fbs`) did not show a statistically significant association. | Chi-Square p-value: 0.761 |
| 8 | Logistic Regression achieved promising classification performance. | Accuracy: 83.6%; F1-score: 84.8% |

**Interpretation:** These results describe associations observed in the analysed dataset. They do not establish medical causation or determine an individual's actual disease risk.

---

## 🤖 Machine Learning Model

### Logistic Regression

A Logistic Regression classification model was developed to predict the target class using the available clinical features.

**Model Development:**
- Train-test split: 80:20
- Stratification: Enabled
- Random state: 42
- Feature scaling: StandardScaler
- Categorical encoding: One-hot encoding
- Classification algorithm: Logistic Regression
- Maximum iterations: 1,000

### Model Performance

The model was evaluated using the following classification metrics:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 83.6% |
| Precision | 84.8% |
| Recall | 84.8% |
| F1-Score | 84.8% |

### Confusion Matrix

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 23 | 5 |
| Actual 1 | 5 | 28 |

**Model Interpretation:**

The Logistic Regression model achieved 83.6% accuracy and an F1-score of 84.8% on the held-out test dataset.

These results demonstrate promising classification performance within this dataset. However, the model requires further validation using larger, independent and representative datasets before any real-world application could be considered.

---

## 🖥️ Interactive Streamlit Application

The project includes an interactive application called **Pulse of Prevention**, developed using Streamlit.

The application demonstrates how machine learning predictions can be presented through a user-friendly interface.

### Application Features

- Custom-designed heart health dashboard.
- Animated heartbeat and ECG header.
- Dataset overview and key performance indicators.
- Cleaned dataset preview.
- Interactive input form covering all 13 original features.
- Logistic Regression-based prediction.
- Prediction probability display.
- Educational disclaimer.

### Application Interface

![Pulse of Prevention - Streamlit Interface](Pulse%20of%20Prevention_User_Interface.png)

### How to Run the Application

**Step 1:** Clone the repository.

```bash
git clone https://github.com/YOUR-USERNAME/pulse-of-prevention-heart-health-analytics.git
```

**Step 2:** Navigate to the application folder.

```bash
cd pulse-of-prevention-heart-health-analytics/Streamlit_App
```

**Step 3:** Install the required libraries.

```bash
pip install streamlit pandas plotly scikit-learn
```

**Step 4:** Run the application.

```bash
streamlit run app.py
```

Alternatively, Windows users can launch the application using the included `Run_App.bat` file.

**Note:** Ensure that `heart_disease.csv` remains in the same folder as `app.py`.

---

## 💡 Business Insights & Recommendations

The analysis demonstrates how data analytics and machine learning can contribute to healthcare research and data-informed decision-making.

### Key Recommendations

| No. | Recommendation | Purpose |
|---|---|---|
| 1 | Monitor important clinical indicators such as chest pain type, maximum heart rate and ST depression. | Support further clinical assessment. |
| 2 | Pay attention to exercise-related symptoms and patterns. | Encourage appropriate medical evaluation. |
| 3 | Use data-driven analysis to identify patterns across different patient groups. | Support healthcare research. |
| 4 | Improve data quality and standardize category codes. | Reduce inconsistencies in future analysis. |
| 5 | Collect larger and more representative datasets. | Improve analytical reliability. |
| 6 | Validate predictive models using independent datasets before considering real-world use. | Reduce the risk of misleading predictions. |

### Business Perspective

This project demonstrates how analytical techniques can help healthcare organizations:

- Explore patterns in patient data.
- Identify variables associated with heart disease.
- Support data-informed healthcare research.
- Develop predictive models for further validation.
- Improve data-driven decision-making.

---

## ⚠️ Project Limitations

The following limitations should be considered when interpreting the results:

1. **Dataset Size:** After removing 723 duplicate occurrences, 302 unique records remained.

2. **Duplicate Uncertainty:** Without patient identifiers, it is not possible to establish whether repeated records represent duplicate patients or repeated observations.

3. **Category Coding:** Some `ca` and `thal` values fall outside the category definitions provided in the accompanying reference PDF.

4. **Limited Features:** The dataset does not include every clinical or lifestyle factor that may influence heart disease.

5. **Model Validation:** Evaluation was performed using a single stratified 80:20 train-test split, resulting in only 61 test records.

6. **Model Limitations:** Logistic Regression may not capture complex nonlinear relationships between clinical variables.

7. **Clinical Limitations:** The model is an educational and analytical demonstration, not a clinically validated diagnostic system.

---

## 🏁 Final Project Conclusion

**Pulse of Prevention: Analyzing Heart Health for Better Outcomes** demonstrates an end-to-end application of Python-based data analytics, statistical hypothesis testing and machine learning.

The analysis explored relationships between clinical indicators and the target variable, identifying several statistically significant associations and distinguishing them from relationships that were not statistically significant.

A Logistic Regression classification model was developed and evaluated, achieving 83.6% accuracy and an 84.8% F1-score on the held-out test dataset.

The project was further extended into an interactive Streamlit application, demonstrating how analytical models can be integrated into accessible user interfaces.

Overall, this project highlights the value of combining exploratory data analysis, statistical reasoning, predictive modelling and business interpretation to transform raw data into meaningful analytical insights.

---

## 👨‍💻 Author

**MD ZIAUL HAQUE**

Data Analyst | Business Analyst

- **LinkedIn:** [Connect with me](https://www.linkedin.com/in/mdziaulhaque-datascience/)
- **GitHub:** [Explore my projects](https://github.com/md-ziaul-haque-ds)
- **Email:** mdziaulhaque.ds@gmail.com

---

## ⚕️ Disclaimer

This project is developed strictly for educational, analytical and portfolio demonstration purposes.

The dataset and machine learning model are not clinically validated. Predictions must not be interpreted as medical diagnoses, treatment recommendations or substitutes for professional medical advice.
