# Heart Disease Prediction App

A Streamlit-based web application for predicting heart disease risk based on patient medical data.

## Live Demo

Try the app on Streamlit: [Heart Disease Prediction App](https://heart-disease-prediction.streamlit.app/)

## Features

- Interactive patient data input form
- Real-time heart disease risk prediction
- Two prediction modes:
  - **Built-in Risk Estimator**: Works without a trained model
  - **ML Model Support**: Automatically loads a trained `.pkl` model if available

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies:
```bash
pip install streamlit numpy
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Using a Trained Model

Place your trained model file in the same directory as `app.py` with one of these names:
- `heart_model.pkl`
- `model.pkl`
- `heart_disease_model.pkl`
- `models/heart_model.pkl`

The app will automatically detect and load it.

## Input Parameters

The app accepts the following patient medical parameters:

- **Age**: Patient age (18-120)
- **Sex**: Male or Female
- **Chest Pain Type**: Typical angina, Atypical angina, Non-anginal pain, or Asymptomatic
- **Resting Blood Pressure**: in mmHg (80-220)
- **Cholesterol**: in mg/dl (100-500)
- **Fasting Blood Sugar**: > 120 mg/dl (Yes/No)
- **Resting ECG**: Normal, ST-T wave abnormality, or Left ventricular hypertrophy
- **Maximum Heart Rate**: achieved (50-220)
- **Exercise-induced Angina**: Yes or No
- **Oldpeak**: ST depression induced by exercise (0-10)
- **Slope of ST Segment**: Upsloping, Flat, or Downsloping
- **Number of Major Vessels**: colored by fluoroscopy (0-4)
- **Thalassemia**: Normal, Fixed defect, Reversible defect, or Unknown

## Output

The app provides:
- Risk classification (High risk / Low risk)
- Probability percentage
- Visual probability indicator

## Project Structure

```
heart/
├── app.py              # Main Streamlit application
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── requirements.txt   # Python dependencies (optional)
```

## License

MIT License

## Author

Your Name

## Support

For issues or questions, please open an issue on GitHub.
