# 🍿 Snack Predictor

Predict your favorite snack using a simple ML model trained on survey responses!

## 🔧 Features
- KNN-based classifier
- Trained on 6 input traits
- Easily connects to Google Sheets

## 📁 Files
- `snack_data.csv` → Your dataset
- `train_model.py` → Trains the model from CSV
- `main.py` → Run predictions
- `requirements.txt` → Python dependencies

## 📦 Setup
1. Clone the repo
2. Add your dataset as `snack_data.csv`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 🚀 Train the Model
```
python train_model.py
```

## 🎯 Predict
```
python main.py
```

## 🗂 Using Google Sheets (Optional)
If your data is in a Google Spreadsheet:
1. Share it with a Google service account (you’ll need a `credentials.json`)
2. Use the `gspread` library:
```python
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

sheet = client.open("YourSheetName").sheet1
data = pd.DataFrame(sheet.get_all_records())
```
3. Save to CSV: `data.to_csv("snack_data.csv", index=False)`

Then continue with training.

---

Designed for high school data science clubs!