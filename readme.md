# Neural Machine Translation: Vietnamese to English with Luong Attention

## 📌 Project Overview
This repository contains a technical implementation of a **Neural Machine Translation (NMT)** system built using PyTorch. The project focuses on translating Vietnamese talk transcripts into English using a Sequence-to-Sequence (Seq2Seq) architecture. 

The implementation demonstrates the transition from a standard LSTM Encoder-Decoder framework to an advanced model utilizing **Luong-style Global Attention**, achieving a significant performance boost in translation quality.

## 📂 Dataset

The model was trained on the IWSLT Evaluation Campaign parallel corpus, which includes English-Vietnamese TED talk transcripts.

## 🚀 Key Technical Features
- **Encoder-Decoder Architecture:** Built a modular NMT framework featuring a contextual encoder to process source sequences and a step-by-step inference decoder.
- **Luong Attention Mechanism:** Implemented a Global Dot-Product Attention layer from first principles to solve the information bottleneck associated with fixed-length context vectors.
- **Dynamic Decoding:** Developed an inference loop that supports both Teacher Forcing (for training efficiency) and recursive decoding (for real-world testing).
- **Standardized Evaluation:** Utilized `SacreBLEU` to compute quantitative metrics, ensuring results are comparable to industry standards.
- **Custom Vocabulary Pipeline:** Integrated a `LanguageDict` class to manage tokenization, numericalization, and handling of special tokens (`<sos>`, `<eos>`, `<pad>`, `<unk>`).

## 🛠 Tech Stack
- **Framework:** PyTorch
- **Evaluation:** SacreBLEU
- **Language:** Python 3.11
- **Processing:** NumPy, Regex

## 📊 Methodology & Results

### 1. Architecture Details
- **Encoder:** A unidirectional LSTM capturing temporal dependencies in the source Vietnamese text.
- **Decoder:** An LSTM-based generator that produces target English tokens.
- **Attention:** Implemented the Dot-Product alignment score to identify relevant source hidden states for every target time step:
  $$score(h_t, \bar{h}_s) = h_t^T \bar{h}_s$$

### 2. Performance Comparison
The impact of the Attention mechanism was evaluated using the BLEU (Bilingual Evaluation Understudy) metric on a held-out test set.

| Model Architecture | BLEU Score | Improvement |
| :--- | :--- | :--- |
| Base Seq2Seq (LSTM) | 4.12 | Baseline |
| **Seq2Seq + Luong Attention** | **14.52** | **+252.4%** |

### 3. Conclusion
The addition of the Attention layer transformed the model's performance. By allowing the decoder to dynamically attend to different parts of the source sentence, the model successfully learned complex cross-lingual alignments that were previously lost in the base LSTM architecture.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aditidankar/NaturalMachineTranslation.git
   ```

2.  Install dependencies:
    ```bash
    pip install torch sacrebleu numpy
    ```

3. Execution:
    Open Vietnamese_English_NMT_Attention.ipynb in a Jupyter environment or Google Colab and run the cells to initiate training and evaluation.




---
This project was developed as part of advanced studies in Neural Networks and Natural Language Processing (EECS7001P) at Queen Mary University of London.
