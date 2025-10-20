# Food_Order_Prediction_DWDM
INTRODUCTION Background: In today’s food industry, restaurants and delivery platforms face challenges in anticipating what customers will order. This unpredictability affects inventory, staff management, and customer satisfaction. Problem Statement: Develop a machine learning model that predicts a customer’s next food order using factors such as past orders, cuisine preferences, time of day, and location. Motivation: • Reduce food waste and manage kitchen inventory efficiently. • Enable faster preparation and smarter staffing. • Deliver personalized recommendations and improve user satisfaction.

PROPOSED SYSTEM Approach: • A supervised ML model (Random Forest) that predicts what food a customer is likely to order based on location, time, past orders, cuisine preference, and order frequency. • The model learns customer patterns to forecast future choices and improve order accuracy. How It’s Better: • Uses multiple real-world features for smarter predictions. • High accuracy and reduced overfitting with Random Forest. • Helps restaurants forecast demand and manage stock. • Provides personalized recommendations for users. • Scalable for integration with food delivery or restaurant systems.

DATASET DESCRIPTION Data Source: Synthetic dataset generated using Python (5,000 records). Features / Attributes: • Location • Time of Day • Previous Order • Preferred Cuisine • Order Frequency • Food Item (Target)

DATA PREPROCESSING

Data Cleaning: • Removed missing values • Fixed inconsistencies
Feature Selection / Encoding: • Label Encoding for categorical columns • StandardScaler for normalization • Split data: 80% Train, 20% Test
MODEL AND ALGORITHM Algorithm Used: Random Forest Classifier Why This Algorithm: • High accuracy for classification • Handles categorical + numerical data well • Avoids overfitting Normalization / Splitting: • Data scaled using StandardScaler • Train-Test split (80-20)

RESULTS AND EVALUATION Accuracy / Metrics: Accuracy: ~90–95% Classification Report & Confusion Matrix Graphs: • Feature Importance Plot • Food Orders by Time/Cuisine/Location Sample Output: Predicted Food Order → “Pizza”

COMPARISON With Other Models: Model Accuracy Decision Tree 86% Logistic Regression 82% Random Forest 94% Performance Summary: Random Forest performed best with balanced precision and recall.

CONCLUSION Every food choice tells a story — and our model learns it. “We built an intelligent system that helps restaurants prepare better, delivery apps recommend smarter, and customers get exactly what they crave — right on time”. Next, we aim to take this from notebooks to the real world — deploying it as a web app powered by real restaurant data.
