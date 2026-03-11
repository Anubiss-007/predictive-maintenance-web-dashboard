import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("--- เริ่มต้นการเตรียมข้อมูล (Data Preprocessing) ---")

# โหลดข้อมูล
df = pd.read_csv('ai4i2020.csv')

# ทำ Data Preprocessing
columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(columns=columns_to_drop)

# แปลงข้อมูลตัวอักษรให้เป็นตัวเลข (Encoding)
# สินค้า Type L (Low), M (Medium), H (High) แปลง ->เป็นตัวเลข 0, 1, 2
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# แยกข้อมูล Features(X) และ Target(y)
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# แบ่งข้อมูลสำหรับสอนโมเดล (Train 80%) และ (Test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("เตรียมข้อมูลเสร็จสิ้น! ขนาดข้อมูล Train:", X_train.shape, "ขนาดข้อมูล Test:", X_test.shape)

print("\n--- เริ่มต้นเทรนโมเดล (Model Training) ---")

# ใช้โมเดล Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predictความแม่นยำด้วยการ Test 20%
y_pred = model.predict(X_test)
print("\nผลการทดสอบความแม่นยำ (Accuracy):", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# บันทึก Model AI เป็นไฟล์ .pkl
joblib.dump(model, 'predictive_model.pkl')
print("\n✅ บันทึกไฟล์ 'predictive_model.pkl' สำเร็จ! พร้อมนำไปใช้งานบน Web App แล้ว")
