
```
# 📝 LSTM Text Generation Model — Alice in Wonderland Dataset

This project implements an end-to-end **LSTM-based text generation model** using **Python** and **PyTorch**.  
The model is trained on the *Alice in Wonderland* dataset sourced from Project Gutenberg and is capable of generating creative English text based on a given input prompt.

---

## 🚀 Features

- ✔ Automatic dataset download from Project Gutenberg  
- ✔ Text cleaning and preprocessing  
- ✔ Character-level encoding  
- ✔ LSTM neural network for next-character prediction  
- ✔ Training loop with epoch-wise loss tracking  
- ✔ Text generation using a custom function  
- ✔ Output saved in `generated_output.txt`

---

## 📂 Project Structure

```

text_generator.py         # Main Python script (model + training + generation)
generated_output.txt      # AI-generated sample text
README.md                 # Project documentation

```

---

## 🧠 Model Architecture

- **Embedding Layer** — converts character IDs to vector embeddings  
- **LSTM Layer (256 units)** — learns sequence patterns and context  
- **Fully Connected Layer** — predicts next character  
- **Loss Function:** CrossEntropy  
- **Optimizer:** Adam (learning rate = 0.003)

This is a **character-level generative language model**.


## 📘 Dataset

- **Source:** Project Gutenberg  
- **Book:** *Alice in Wonderland*  
- Dataset is downloaded automatically using Python’s `requests` library.  
- Text is cleaned and converted to lowercase with punctuation filtering.


## ▶ How to Run the Project

### 1️⃣ Clone this repository

```

git clone [https://github.com/USERNAME/text-generation-lstm](https://github.com/USERNAME/text-generation-lstm)
cd text-generation-lstm

```

### 2️⃣ Create virtual environment

```

python -m venv venv
venv\Scripts\activate     # Windows

```

### 3️⃣ Install required libraries

```

pip install torch numpy requests

```

### 4️⃣ Run the model

```

python text_generator.py

```

Training takes **6–8 minutes**, depending on your CPU.

---

## 📄 Output

- The generated text is saved automatically in:

```

generated_output.txt

```

Example output:

```

alice was wandering through the forest when suddenly she saw...

```

---

## 🎯 Purpose of the Project

This project demonstrates the core concepts of:

- Data preprocessing  
- Sequence modeling  
- Neural text generation  
- LSTM architecture  
- Character-level AI models  
- End-to-end Python ML workflow  

It fulfills the requirements of a **Text Generation Assignment** for ML/AI internships or coding evaluations.

---


