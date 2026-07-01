import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Task2_Credit_Card_Fraud_Detection/creditcard.csv')

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

le_dict = {}
categorical_cols = ['merchant', 'category', 'gender', 'job']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

X = df.drop('is_fraud', axis=1) if 'is_fraud' in df.columns else df.drop('Class', axis=1)
y = df['is_fraud'] if 'is_fraud' in df.columns else df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=15, random_state=42, max_depth=8, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

def check_fraud():
    try:
        amt = float(amt_entry.get())
        city_pop = float(pop_entry.get())
        
        input_data = pd.DataFrame([[0, 0, 0, amt, 0, 0, city_pop, 0, 0, 0, 0]], columns=X.columns)
        prediction = model.predict(input_data)[0]
        
        if prediction == 1 or amt > 5000:
            messagebox.showwarning("Result", "Warning: High Risk of Fraudulent Transaction!")
        else:
            messagebox.showinfo("Result", "Success: Transaction is Safe and Genuine.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

root = tk.Tk()
root.title("Credit Card Fraud Detection System")
root.geometry("450x320")
root.resizable(False, False)
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use('default')
style.configure("TFrame", background="#1e1e1e")
style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Helvetica", 10))
style.configure("TEntry", fieldbackground="#2d2d2d", foreground="#ffffff", insertcolor="white")
style.configure("TButton", background="#007acc", foreground="#ffffff", font=("Helvetica", 10, "bold"))
style.map("TButton", background=[('active', '#005999')])

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(main_frame, text="Transaction Fraud Assessment", font=("Helvetica", 14, "bold"), foreground="#007acc")
title_label.pack(pady=10)

amt_label = ttk.Label(main_frame, text="Transaction Amount ($):")
amt_label.pack(anchor=tk.W, pady=2)
amt_entry = ttk.Entry(main_frame, width=40)
amt_entry.pack(fill=tk.X, pady=5)

pop_label = ttk.Label(main_frame, text="City Population:")
pop_label.pack(anchor=tk.W, pady=2)
pop_entry = ttk.Entry(main_frame, width=40)
pop_entry.pack(fill=tk.X, pady=5)

predict_btn = ttk.Button(main_frame, text="Analyze Transaction", command=check_fraud)
predict_btn.pack(pady=25)

root.mainloop()
