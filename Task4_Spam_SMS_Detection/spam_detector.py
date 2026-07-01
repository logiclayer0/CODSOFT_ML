import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Spam SMS Detection Engine")
        self.root.geometry("600x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Helvetica", 10))
        style.configure("TButton", background="#007acc", foreground="#ffffff", font=("Helvetica", 10, "bold"))
        style.map("TButton", background=[('active', '#005999')])
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.label_title = ttk.Label(main_frame, text="SMS Spam Classification Engine", font=("Helvetica", 16, "bold"), foreground="#007acc")
        self.label_title.pack(pady=15)
        
        self.label_status = ttk.Label(main_frame, text="Status: Initializing system...", font=("Helvetica", 10, "italic"), foreground="#aaaaaa")
        self.label_status.pack(pady=5)
        
        self.text_msg = tk.Text(main_frame, height=10, width=65, font=("Helvetica", 10), bg="#2d2d2d", fg="#ffffff", insertbackground="white", relief=tk.FLAT)
        self.text_msg.pack(pady=15)
        self.text_msg.config(state=tk.DISABLED)
        
        self.btn_predict = ttk.Button(main_frame, text="Analyze Message", command=self.predict_spam, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)
        
        self.label_result = ttk.Label(main_frame, text="", font=("Helvetica", 14, "bold"))
        self.label_result.pack(pady=20)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = MultinomialNB(alpha=0.1)
        
        self.root.after(100, self.initialize_ml_engine)

    def initialize_ml_engine(self):
        train_file = 'spam.csv'
        if not os.path.exists(train_file):
            train_file = 'Task4_Spam_SMS_Detection/spam.csv'
            
        if not os.path.exists(train_file):
            self.label_status.config(text="Error: Dataset missing.", foreground="#ff3333")
            messagebox.showerror("Data Error", "Required spam.csv dataset file not found.")
            return
            
        self.label_status.config(text="Status: Loading and preprocessing dataset...")
        self.root.update_idletasks()
        
        try:
            df = pd.read_csv(train_file, encoding='latin-1')
        except:
            df = pd.read_csv(train_file, encoding='utf-8')
            
        df = df.iloc[:, [0, 1]]
        df.columns = ['v1', 'v2']
        df = df.dropna()
        
        self.label_status.config(text="Status: Training Machine Learning model...")
        self.root.update_idletasks()
        
        X_train = self.vectorizer.fit_transform(df['v2'])
        y_train = df['v1']
        self.model.fit(X_train, y_train)
        
        self.label_status.config(text="Status: System Ready", foreground="#4caf50")
        self.text_msg.config(state=tk.NORMAL)
        self.btn_predict.config(state=tk.NORMAL)

    def predict_spam(self):
        user_input = self.text_msg.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a valid text message to analyze.")
            return
            
        vectorized_input = self.vectorizer.transform([user_input])
        prediction = self.model.predict(vectorized_input)[0]
        
        if prediction.lower() == 'spam':
            self.label_result.config(text="RESULT: SPAM / FRAUD DETECTED!", foreground="#ff3333")
            messagebox.showwarning("Security Alert", "Warning: This message shows high indicators of spam/phishing.")
        else:
            self.label_result.config(text="RESULT: HAM (SAFE MESSAGE)", foreground="#4caf50")
            messagebox.showinfo("Analysis Result", "Success: This message is safe and clean.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()