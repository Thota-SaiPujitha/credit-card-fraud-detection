1.Credit Card Fraud Detection
Machine-learning based system to detect fraudulent credit-card transactions.

 Features
- Data preprocessing & class balancing (SMOTE)  
- Dimensionality reduction with PCA  
- Models: Logistic Regression, Random Forest, XGBoost  
- Interactive Streamlit dashboard for live transaction prediction  
- Visualizations of feature importances and performance metrics

Folder Structure
├── app/ # Streamlit dashboard code
├── model/ # Training & evaluation scripts
├── utils/ # Helper functions (data loading, preprocessing)
├── visualisations/ # Plotting scripts
├── README.md # Project overview & instructions
├── requirements.txt # Python dependencies
└── .gitignore # Files/folders to exclude from Git

Getting Started
 **Clone the repo**  
   git clone https://github.com/USERNAME/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2.Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

3.Install dependencies

pip install -r requirements.txt
4.Run the Streamlit app
streamlit run app/main.py
5.Reproduce model training
python model/train.py
6.Results
Logistic Regression: ROC AUC = 0.92
Random Forest: ROC AUC = 0.96
XGBoost: ROC AUC = 0.97
See visualisations/ for detailed plots.

