# Burn Study Logistic Regression Model

## Project Overview

This project developed a logistic regression model to predict patient survival after burn injuries.

## Business Problem

Burn injuries cause significant mortality worldwide. The objective was to identify which factors most strongly influence survival outcomes.

## Dataset

- 1000 records
- 9 variables
- Burn patient records

## Methods

- Exploratory Data Analysis
- Data Cleaning
- Feature Analysis
- Logistic Regression
- Model Evaluation

## Key Findings

### Strongest Predictors

1. Total Burn Surface Area (TBSA)
2. Age

The model achieved approximately 98-99% predictive accuracy.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Visualizations

### Tbsa Boxplot

The boxplot relates the total burn surface area with death, showing that most of the death cases had around 20-65% of surface burnt, which brings the interpretation that all the deaths had a considerable amount of the body burnt, meaning that this could had been decisive for the outcome.

<img width="288" height="224" alt="Picture4" src="https://github.com/user-attachments/assets/33f8dba5-7079-4413-925e-26e28d084631" />

### Heatmap showing correlations

The heatmap presents the relationship between age, total burn surface area, and death, where the closer to red would be the variables that are mostly related. In this case, the strongest relationship was between total burn surface area and death, as also presented on a previous boxplot visualization. And there is some relation between age and death as well. 

<img width="374" height="270" alt="Picture1" src="https://github.com/user-attachments/assets/c620fbfc-f078-4319-aa33-1cfef5c96486" />

### Logistic Regrassion Sigmoid Curve

With the goal to find the probability of death based on the dataset presented, the logistic function was used to create a sigmoid visualization that presents the relation between probability and z (linear score), where the more negative the z more likely would be a case of survival from burn injury, while the higher the z also the higher the possibility of death from burnt. And if the probability is 0.5 or higher, it predicts death, and on the other hand, if it is lower than 0.5, then it predicts survival, as can be seen in the visualization below.

<img width="358" height="278" alt="Picture5" src="https://github.com/user-attachments/assets/e6fae451-b4e5-40f6-93f0-17251d5fe3a7" />

### Confusion Matrix

The confusion matrix compares the actual and predicted results, confirming that the majority were true positives, followed by true negatives.

<img width="316" height="268" alt="Picture2" src="https://github.com/user-attachments/assets/6734ad33-2028-4009-8c3d-0550b1376942" />

### ROC Curve

The ROC curve also confirms that the model can almost always predict results correctly since the AUC line goes around the graph meeting both ends from the true and false positive predictions.

<img width="306" height="245" alt="Picture3" src="https://github.com/user-attachments/assets/89554aec-8f45-482c-8e43-a2288b133e70" />

## Files

[Python code.py](https://github.com/user-attachments/files/28562866/Python.code.py)

[Logistic Regression Model.docx](https://github.com/user-attachments/files/28562907/Logistic.Regression.Model.docx)

