# 🧠 Employee Attrition Analysis & Agentic AI System

A full-stack data analytics project that predicts employee attrition using Machine Learning and flags high-risk employees using a Rule-Based Agentic AI system — built with SQL, Python, and Tableau.

---

## 📌 Project Overview

Employee turnover is one of the most costly challenges faced by organizations today. This project builds an end-to-end pipeline that:

- Analyses employee data using **SQL**
- Predicts attrition using a **Random Forest ML model**
- Automatically flags at-risk employees using a **Rule-Based Agentic AI**
- Visualises insights through an interactive **Tableau Dashboard**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| SQLite | Data storage and SQL analysis |
| Python (Pandas, Scikit-learn) | Data processing and ML model |
| Random Forest Classifier | Attrition prediction |
| Rule-Based Agentic AI | Automated HR intervention recommendations |
| Tableau Public | Interactive dashboard |

---

## 📂 Project Structure

```
employee-attrition-analysis/
│
├── attrition_agent.py        # Main Python script (ML + Agentic AI)
├── enriched_employees2.csv   # Enriched dataset with Risk Scores
├── attrition_results.csv     # Final output with predictions & recommendations
└── README.md                 # Project documentation
```

---

## 📊 Dataset

- **Name:** IBM HR Analytics Employee Attrition & Performance
- **Source:** [Kaggle - IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Created by:** IBM Data Scientists (Synthetic Dataset)
- **Size:** 1,470 employees, 35 original features
- **Target Variable:** Attrition (Yes = Left, No = Stayed)
- **Class Distribution:** ~84% No (1,233 employees), ~16% Yes (237 employees)

### 📋 Key Features in the Dataset

| Feature | Description |
|---------|-------------|
| Age | Employee age (18–60) |
| Attrition | Whether the employee left (Yes/No) — Target Variable |
| BusinessTravel | Frequency of business travel (Non-Travel, Travel_Rarely, Travel_Frequently) |
| Department | Department (Sales, Research & Development, Human Resources) |
| DistanceFromHome | Distance from home to office (in km) |
| Education | Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor) |
| EducationField | Field of education (Life Sciences, Medical, Marketing, etc.) |
| EnvironmentSatisfaction | Satisfaction with work environment (1=Low, 4=Very High) |
| Gender | Male / Female |
| JobInvolvement | Level of job involvement (1=Low, 4=Very High) |
| JobLevel | Job level in hierarchy (1–5) |
| JobRole | Role in the company (Sales Executive, Research Scientist, etc.) |
| JobSatisfaction | Satisfaction with job (1=Low, 4=Very High) |
| MaritalStatus | Single, Married, Divorced |
| MonthlyIncome | Monthly salary in USD |
| NumCompaniesWorked | Number of companies worked at previously |
| OverTime | Whether employee works overtime (Yes/No) |
| PercentSalaryHike | Percentage salary increase last year |
| PerformanceRating | Last performance rating (1=Low, 4=Outstanding) |
| RelationshipSatisfaction | Satisfaction with relationships at work (1=Low, 4=Very High) |
| StockOptionLevel | Stock option level (0–3) |
| TotalWorkingYears | Total years of work experience |
| TrainingTimesLastYear | Number of training sessions attended last year |
| WorkLifeBalance | Work-life balance rating (1=Bad, 4=Best) |
| YearsAtCompany | Years spent at current company |
| YearsInCurrentRole | Years in current role |
| YearsSinceLastPromotion | Years since last promotion |
| YearsWithCurrManager | Years with current manager |

### 🔧 Engineered Features (Added via SQL)

| Feature | Description |
|---------|-------------|
| RiskScore | Composite risk score calculated from multiple attrition factors |
| AgeGroup | Age grouped into bands (Under 25, 25-34, 35-44, 45-54, 55+) |
| SalaryBand | Salary grouped into bands (Low, Mid, High, Very High) |
| TenureGroup | Years at company grouped (0-1 Years, 2-5 Years, 6-10 Years, 10+ Years) |

> ⚠️ Note: This is a **synthetic dataset** created by IBM for educational purposes. It does not represent real employee data.

---

## ⚙️ How It Works

### 1. SQL Analysis
The raw dataset was imported into SQLite and analysed using SQL queries to extract key insights:
- Attrition rate by Department, Job Role, Age Group
- Impact of Overtime, Salary, and Work-Life Balance
- Risk score calculation per employee

### 2. Machine Learning Model
A **Random Forest Classifier** was trained to predict the probability of an employee leaving:
- 32 features used for training
- 75/25 train-test split
- Class balancing to handle imbalanced attrition data
- Output: Attrition probability score per employee

### 3. Agentic AI System
A rule-based agent autonomously:
- Evaluates each employee across 13 risk dimensions
- Assigns a Risk Tier: 🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low
- Generates personalised HR intervention recommendations
- Exports a full HR report to CSV

### 4. Tableau Dashboard
An interactive dashboard visualising:
- Key KPIs (Total Employees, Attrition Rate, Critical Risk Count)
- Attrition by Department, Job Role, Age Group, Tenure
- Overtime Impact analysis
- Risk Score Heatmap by Department and Job Role

---

## 📈 Key Findings

- Overall attrition rate: **16.12%**
- **Sales Representatives** have the highest attrition rate at ~40%
- Employees working **overtime** are significantly more likely to leave
- **225 employees** identified as Critical Risk by the Agentic AI
- Top attrition drivers: Monthly Income, Overtime, Job Satisfaction, Years at Company

---

## 🤖 Agentic AI Risk Flags

The agent evaluates employees across these dimensions:

| Flag | Condition |
|------|-----------|
| LOW_SALARY | Monthly Income < $3,000 |
| OVERTIME_STRESS | Employee works overtime |
| POOR_WLB | Work-Life Balance score = 1 |
| LOW_JOB_SATISFACTION | Job Satisfaction score = 1 |
| STAGNANT_CAREER | No promotion in 4+ years |
| NEW_HIRE_FLIGHT_RISK | Less than 2 years at company |
| HIGH_JOB_HOPPER | Worked at 5+ companies |
| ML_HIGH_RISK | ML model predicts 70%+ attrition probability |

---

## 📊 Tableau Dashboard

View the live dashboard here:
[Employee Attrition Analysis Dashboard](https://public.tableau.com/app/profile/saniya.krishnaraj)

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/krishsanii28-droid/employee-attrition-analysis.git
```

2. Install dependencies:
```bash
pip3 install pandas scikit-learn
```

3. Run the agent:
```bash
python3 attrition_agent.py
```

4. Check the output in `attrition_results.csv`

---

## 👩‍💻 Author

**Saniya Krishnaraj**
B.Tech Computer Science & Engineering with Business Systems
Honours in Machine Learning
