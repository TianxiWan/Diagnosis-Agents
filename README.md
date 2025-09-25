# Diagnosis-Agents  
Implementation of “From Medical Records to Diagnostic Dialogues: A Clinically-Grounded Framework for Psychiatric Comorbidity”  
*Research paper under review.*

---


## 1. Mandatory Edits  

Open `llm_tools_api.py` and fill in the corresponding API key and base URL based on the model you intend to use.

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
```

## 📁 Input Format and Examples

We provide a complete sample EMR in `raw_data/cases_completed.json`, which includes both the personal history dictionary and the fictitious experience dictionary. If you wish to use new EMR data, make sure the structure strictly follows this format.

Additionally, we include a sample dialogue output in the `Dial_data/` folder to demonstrate the expected format of the generated conversations.

# 📊 PsycoData Datasets  

The EMR dataset **PsyCoProfile** and the diagnostic dialogue dataset **PsyCoTalk** have successfully passed institutional ethics review. Both are available in this repo under `PsyCoData/`.  

PsycoData is the **first large-scale, clinically standardised resource for psychiatric *comorbidity***, consisting of two complementary parts:

| File | Records | Description |
|------|---------|-------------|
| **`PsyCoProfile.json`** | 502 | Structured EMRs covering six frequent combinations of four core disorders: **MDD, AD, BD, ADHD**. Each EMR includes five *personal histories* and ten *fictitious experiences* for augmentation. |
| **`PsyCoTalk.json`** | 3,000 | Multi-turn diagnostic dialogues generated from EMRs via a **multi-agent simulator guided by a Hierarchical Diagnostic State Machine (HDSM)**. Dialogues average **45.9 turns**, with mean lengths of **34.0 words (doctor)** and **43.5 words (patient)**. |

These datasets enable training and evaluation of LLMs that must screen **multiple psychiatric disorders jointly** and reason over realistic clinical workflows.

---

## Dataset Creation  

- **PsyCoProfile**: Synthesized EMRs capturing diverse patient profiles and symptom presentations.  
- **PsyCoTalk**: Simulated dialogues that emulate clinical interviews, balancing *clinical consistency* with *linguistic naturalness*.  

Both resources have been reviewed by licensed psychiatrists for validity and plausibility.

---

## Dataset Statistics  

### PsyCoProfile  
- **Total EMRs**: 502  
- **Avg. posts per user**: 134  
- **Avg. symptom posts per user**: 25  
- **Avg. life event posts per user**: 13  
- **Avg. distinct symptoms per user**: 27  

### PsyCoTalk  
- **Total dialogues**: 3,000  
- **Avg. turns per dialogue**: 45.9  
- **Avg. doctor utterance length**: 34.0 words  
- **Avg. patient utterance length**: 43.5 words  

---

## Uses  

### Direct Use  
- Training and evaluation of multi-label psychiatric diagnostic models  
- Building dialogue systems for **comorbid disorder screening**  
- Research on conversational patterns in psychiatric consultations  

### Out-of-Scope Use  
- Deployment in real-world clinical settings without further validation.
- Use in contexts requiring real patient data, as the datasets are synthetic.

---

## Ethical Considerations  

- **Data Privacy**: All data are synthetic and do not contain personally identifiable information.
- **Clinical Use**: The datasets are intended for research purposes only and should not be used for clinical decision-making without appropriate validation.
