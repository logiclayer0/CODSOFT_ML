import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class MovieGenreApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enterprise Movie Genre Classifier")
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
        
        self.label_title = ttk.Label(main_frame, text="Movie Genre Classification Engine", font=("Helvetica", 16, "bold"), foreground="#007acc")
        self.label_title.pack(pady=15)
        
        self.label_status = ttk.Label(main_frame, text="Status: Initializing system...", font=("Helvetica", 10, "italic"), foreground="#aaaaaa")
        self.label_status.pack(pady=5)
        
        self.text_plot = tk.Text(main_frame, height=10, width=65, font=("Helvetica", 10), bg="#2d2d2d", fg="#ffffff", insertbackground="white", relief=tk.FLAT)
        self.text_plot.pack(pady=15)
        self.text_plot.config(state=tk.DISABLED)
        
        self.btn_predict = ttk.Button(main_frame, text="Predict Genre", command=self.predict_genre, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)
        
        self.label_result = ttk.Label(main_frame, text="", font=("Helvetica", 14, "bold"), foreground="#4caf50")
        self.label_result.pack(pady=20)
        
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.model = MultinomialNB(alpha=0.1)
        
        self.root.after(100, self.initialize_ml_engine)

    def initialize_ml_engine(self):
        train_file = 'train_data.txt'
        if not os.path.exists(train_file):
            train_file = 'Task1_Movie_Genre/train_data.txt'
            
        if not os.path.exists(train_file):
            self.label_status.config(text="Error: Dataset files missing.", foreground="#ff3333")
            messagebox.showerror("Data Error", f"Required training dataset file not found.")
            return
            
        self.label_status.config(text="Status: Loading training dataset...")
        self.root.update_idletasks()
        
        lines = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':::')
                if len(parts) >= 4:
                    lines.append([parts[2].strip(), parts[3].strip()])
        df = pd.DataFrame(lines, columns=['Genre', 'Description'])
        
        self.label_status.config(text="Status: Training Machine Learning model...")
        self.root.update_idletasks()
        
        X_train = self.vectorizer.fit_transform(df['Description'])
        y_train = df['Genre']
        self.model.fit(X_train, y_train)
        
        self.label_status.config(text="Status: System Ready", foreground="#4caf50")
        self.text_plot.config(state=tk.NORMAL)
        self.btn_predict.config(state=tk.NORMAL)

    def predict_genre(self):
        user_input = self.text_plot.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a valid movie plot summary.")
            return
            
        vectorized_input = self.vectorizer.transform([user_input])
        prediction = self.model.predict(vectorized_input)
        self.label_result.config(text=f"PREDICTED GENRE: {prediction[0].upper()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieGenreApp(root)
    root.mainloop()