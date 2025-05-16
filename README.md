# PsyCo-Agents  
Official implementation for our NeurIPS 2025 paper  
**“Two-Stage Construction of Structured Comorbidity EMRs and Diagnostic Dialogues”**

---

## ⚠️ Update  
The PsyCo-Dial conversation set is under institutional ethics review.  
The files required to reproduce the pipeline are available now; the full dataset will appear here immediately after approval.

---

## 1. Mandatory Edits  

Open `llm_tools_api.py`:

| Line   | Purpose       | What to put in                                               |
|--------|---------------|--------------------------------------------------------------|
| 53–54  | GPT keys      | Your `OPENAI_API_KEY` and `OPENAI_API_BASE`                 |
| 60–61  | Qwen keys     | Your `QWEN_API_KEY` and `QWEN_API_BASE`                     |
| 71–72  | DeepSeek keys | Your `DEEPSEEK_API_KEY` and `DEEPSEEK_API_BASE`             |

Open `patient_template_gen.py`:

- Line 12: `MODELNAME` ← LLM used to create fictitious patient experiences  

Open `main.py`:

- Line 18: `MODEL_NAME` ← LLM used to generate the final dialogues  

---

## 2. Optional Edits  

In `patient_template_gen.py`:

- Line 13: `PATIENT_COUNT` — Number of EMRs to use  
- Line 14: `FicExp_COUNT` — Fictitious experiences per EMR  

In `main.py`:

- `NUM` — Conversations per fictitious experience  

Defaults are all set to 1 for a quick test run.

---

## 3. Run the Pipeline  

```bash
# Step 1: build dialogue-ready cases and fictitious experience descriptions
python patient_template_gen.py  

# Step 2: generate multi-turn diagnostic conversations
python main.py
