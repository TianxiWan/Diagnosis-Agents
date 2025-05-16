# HDSM-Agents  
Implementation of “From Medical Records to Diagnostic Dialogues: A Clinically-Grounded Framework for Psychiatric Comorbidity”  
*Research paper under review.*

---

## ⚠️ Update  
The EMR dataset **PsyCoProfile** and the diagnostic dialogue dataset **PsyCoTalk** have successfully passed institutional ethics review and are now available on Hugging Face.  
You can access them at: [https://huggingface.co/datasets/TianXiWan/PsycoData](https://huggingface.co/datasets/TianXiWan/PsycoData)


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


