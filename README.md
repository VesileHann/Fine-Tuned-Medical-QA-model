# Fine-Tuned-Medical-QA-model

```
# Fine-Tuned Medical QA Model  

## Overview  
This repository contains a fine-tuned version of EleutherAI's Pythia-410m model for medical question answering. The model is trained on [Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) to generate doctor-style responses to patient queries.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/VesileHann/medical-qa-model.git  
   ```  
2. Install dependencies:  
   ```bash  
   pip install transformers datasets torch  
   ```  

## Usage  
### Training  
Run the provided code to fine-tune the model:  
```python  
# Refer to the code in the repository for full training details  
```  

### Inference  
Example usage for generating responses:  
```python  
from transformers import AutoModelForCausalLM, AutoTokenizer  

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_EleutherAI_medical_model")  
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_EleutherAI_medical_model")  

prompt = """Act as a doctor. Keep annotations concise.  
Question: [Your patient query here]  
Answer: """  

# Tokenize and generate response  
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
outputs = model.generate(  
    inputs.input_ids,  
    max_new_tokens=200,  
    temperature=0.7,  
    top_p=0.9,  
    do_sample=True,  
    pad_token_id=tokenizer.eos_token_id  
)  
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)  
print("Generated:", response)  
```  

## Dataset  
The model is trained on a processed subset of the medical QA dataset, containing 246,678 examples. Columns include `instruction`, `input` (patient query), and `output` (doctor response).  

## Results  
- **Training Loss**: 0.64  
- **Evaluation Loss**: 0.59  

## Contributing  
Pull requests are welcome! For major changes, open an issue first.  
