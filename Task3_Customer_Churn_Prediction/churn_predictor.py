import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ChurnPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Churn Prediction Engine")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Helvetica", 10))
        style.configure("TEntry", fieldbackground="#2d2d2d", foreground="#ffffff", insertcolor="white")
        style.configure("TButton", background="#007acc", foreground="#ffffff", font=("Helvetica", 10, "bold"))
        style.map("TButton", background=[('active', '#005999')])
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.label_title = ttk.Label(main_frame, text="Customer Churn Analysis", font=("Helvetica", 16, "bold"), foreground="#007acc")
        self.label_title.pack(pady=15)
        
        self.label_status = ttk.Label(main_frame, text="Status: Initializing system...", font=("Helvetica", 10, "italic"), foreground="#aaaaaa")
        self.label_status.pack(pady=5)
        
        self.credit_label = ttk.Label(main_frame, text="Credit Score:")
        self.credit_label.pack(anchor=tk.W, pady=2)
        self.credit_entry = ttk.Entry(main_frame, width=45)
        self.credit_entry.pack(fill=tk.X, pady=5)
        
        self.age_label = ttk.Label(main_frame, text="Age:")
        self.age_label.pack(anchor=tk.W, pady=2)
        self.age_entry = ttk.Entry(main_frame, width=45)
        self.age_entry.pack(fill=tk.X, pady=5)
        
        self.balance_label = ttk.Label(main_frame, text="Account Balance ($):")
        self.balance_label.pack(anchor=tk.W, pady=2)
        self.balance_entry = ttk.Entry(main_frame, width=45)
        self.balance_entry.pack(fill=tk.X, pady=5)
        
        self.btn_predict = ttk.Button(main_frame, text="Analyze Churn Risk", command=self.predict_churn, state=tk.DISABLED)
        self.btn_predict.pack(pady=25)
        
        self.root.after(100, self.initialize_ml_engine)

    def initialize_ml_engine(self):
        train_file = 'churn.csv'
        if not os.path.exists(train_file):
            train_file = 'Task3_Customer_Churn_Prediction/churn.csv'
            
        if not os.path.exists(train_file):
            self.label_status.config(text="Error: Dataset missing.", foreground="#ff3333")
            messagebox.showerror("Data Error", "Required churn.csv dataset file not found.")
            return
            
        self.label_status.config(text="Status: Training Machine Learning model...")
        self.root.update_idletasks()
        
        df = pd.read_csv(train_file)
        
        drop_cols = ['RowNumber', 'CustomerId', 'Surname']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        le = LabelEncoder()
        for col in ['Geography', 'Gender']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))
                
        target_col = 'Exited' if 'Exited' in df.columns else df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_columns = list(X.columns)
        self.mean_values = X.mean()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        self.label_status.config(text="Status: System Ready", foreground="#4caf50")
        self.btn_predict.config(state=tk.NORMAL)

    def predict_churn(self):
        try:
            credit_score = float(self.credit_entry.get())
            age = float(self.age_entry.get())
            balance = float(self.balance_entry.get())
            
            if credit_score <= 500 or age >= 60 or balance >= 100000:
                messagebox.showwarning("Result", "Warning: High Risk! Customer is likely to leave.")
                return

            input_row = self.mean_values.copy()
            for col in self.feature_columns:
                if 'credit' in col.lower() or 'score' in col.lower():
                    input_row[col] = credit_score
                elif 'age' in col.lower():
                    input_row[col] = age
                elif 'balance' in col.lower():
                    input_row[col] = balance
            
            input_df = pd.DataFrame([input_row], columns=self.feature_columns)
            prediction = self.model.predict(input_df)[0]
            
            if prediction == 1:
                messagebox.showwarning("Result", "Warning: High Risk! Customer is likely to leave.")
            else:
                messagebox.showinfo("Result", "Success: Customer is loyal and safe.")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictorApp(root)
    root.mainloop()