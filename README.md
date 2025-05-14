# 🧪 RDF & SHACL Generator + Validator + Ontology Visualizer

This Streamlit application is a tool for materials scientists and semantic web developers who need to convert material test data into structured RDF (Resource Description Framework) and SHACL (Shapes Constraint Language) models. The app offers a robust pipeline enhanced by LLM-powered correction and explanation logic to improve data quality and semantic modeling.

[View live website here](https://llm-rdf-shacl-creation-5623b0cfb7c0.herokuapp.com/)

---

## 🚀 Key Features

- **Multi-LLM Support**
  Choose between:

  - **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo
  - **Anthropic Claude**: Opus, Sonnet, Haiku
  - **Ollama (Local)**: Self-hosted models like LLaMA3, Qwen, Phi

- **RDF Generation**
  Converts raw mechanical test data (e.g., creep test reports) into structured RDF in Turtle syntax.

- **SHACL Shape Creation**
  Automatically generates SHACL validation shapes to enforce semantic constraints.

- **🔁 Auto-Correction Loop (New!)**
  The LLM improves its RDF/SHACL outputs through a 3-step pipeline:

  1. Initial generation
  2. First refinement
  3. Final refinement

- **🧠 Critique & Explanation (New!)**
  After each RDF/SHACL version, the LLM explains:

  - What could be improved
  - Why semantic or structural changes are needed
    This is shown to the user before the next version is generated.

- **⚡ Progressive Display (New!)**
  Each version (initial, optimized 1, optimized 2) is shown **as soon as it's generated**, with side-by-side improvement explanations.

- **✅ Validation Engine**
  Automatically validates the final RDF against the SHACL shapes using `pySHACL`.

- **🌐 Interactive Visualization**
  Renders RDF graphs using Pyvis & NetworkX for intuitive visual exploration.

- **📚 Ontology Term Suggestion**
  The app suggests relevant ontology classes/properties from:

  - EMMO
  - MATWERK
  - PMDcore
  - NFDI Core
  - IAO, OBI, OBO
  - QUDT

- **⬇️ Export Functionality**
  Easily download the final RDF and SHACL files for reuse and integration.

---

## 🛠 Technologies Used

- **Streamlit** – UI framework
- **OpenAI API / Anthropic Claude API / Ollama** – LLM generation
- **RDFlib** – RDF parsing and graph construction
- **PySHACL** – RDF & SHACL validation engine
- **NetworkX + Pyvis** – Graph visualization
- **dotenv** – Secure environment variable handling
- **Streamlit Components** – HTML embedding
- **Temporary Files** – Visualization output caching

---

## ✨ Prompt Engineering Highlights

- **Domain-Specific Prompts**
  Encodes materials science ontology knowledge and semantic web best practices.

- **Multi-Stage Generation**
  Separate stages for RDF, SHACL, and ontology suggestions.

- **Correction + Explanation Logic (New!)**
  Prompts include:

  - Structured critique of outputs
  - Self-refinement based on critique
  - Instruction-following behavior tuned for consistency

- **Output Control**
  Uses formatting constraints, temperature settings, and parsing structure to maintain valid and interoperable RDF data.

---

## 🧪 Use Cases

This tool is particularly useful for:

- Materials scientists working with **experimental test data**
- Researchers building **knowledge graphs or semantic databases**
- FAIR data advocates ensuring **interoperability and quality**
- Engineers needing **automated SHACL validation**
- Anyone transforming **raw text to semantic web standards**

---

## 🧩 How It Works

1. **Paste mechanical test data** or use example input.
2. **Choose an LLM provider and model** (OpenAI, Claude, Ollama).
3. Click "Generate" — the app:

   - Produces the first RDF/SHACL
   - Critiques it
   - Improves it twice
   - Validates and visualizes the final result

---

## 📦 Future Enhancements

- User-editable correction steps
- Persistent history with version comparison
- RDF triple explanation layer
- Graph editing and export to RDF

---
