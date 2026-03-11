# ⚙️ AI Predictive Maintenance Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-maintenance-ml-web-dashboard.streamlit.app/)

<img width="1490" height="818" alt="Screenshot 2569-03-11 at 16 45 30" src="https://github.com/user-attachments/assets/880a1370-6692-46c5-99d0-0e243da95427" />

<img width="1490" height="892" alt="Screenshot 2569-03-11 at 16 46 08" src="https://github.com/user-attachments/assets/2633f6d6-1fdb-4668-842d-d4a6b7fa6a44" />

## 📌 Project Overview
An interactive Machine Learning web application designed to predict manufacturing equipment failures using real-time sensor data. This project aligns with **Lean Manufacturing** principles by enabling early detection of equipment anomalies, minimizing unexpected downtime, and optimizing maintenance schedules.

## ✨ Key Features
* **Real-time Predictive Analytics:** Utilizes a pre-trained **Random Forest Classifier** to analyze sensor inputs (Temperature, RPM, Torque, Tool Wear) and predict failure probabilities.
* **Root Cause Analysis:** Visualizes feature importance to identify the primary factors contributing to potential machine breakdowns.
* **Prescriptive Action Plan:** Goes beyond prediction by automatically generating actionable maintenance recommendations (e.g., Tool Replacement, Motor Load Check) based on specific risk thresholds.
* **Exportable Reports:** Allows engineers to download the diagnostic results and action plans as a clean CSV file for work order generation.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn (Random Forest)
* **Data Manipulation:** Pandas
* **Data Visualization:** Plotly
* **Web Framework:** Streamlit
