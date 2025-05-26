import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import re
import requests
from dotenv import load_dotenv
load_dotenv()

# # Initialize clients
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
st.set_page_config(page_title="RDF & SHACL Generator Using LLM", layout="wide")
st.title("🔬 RDF & SHACL Generator + Validator + Ontology Visualizer Using LLM")

# Add API key input fields to the sidebar
st.sidebar.header("API Keys")

st.sidebar.markdown("**OpenAI API Key**")
st.sidebar.markdown("Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.markdown("**Anthropic API Key**")
st.sidebar.markdown("Get your Anthropic API key from [Anthropic Console](https://www.merge.dev/blog/anthropic-api-key)")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")

# Add Ollama endpoint configuration
st.sidebar.markdown("**Ollama Configuration**")
st.sidebar.markdown("Enter your Ollama API endpoint")
ollama_endpoint = st.sidebar.text_input("Ollama API Endpoint", value="http://localhost:11434")

# Add temperature settings - will be shown conditionally
temperature_values = {
    "OpenAI": 0.3,
    "Anthropic (Claude)": 0.3,
    "Ollama (Self-hosted)": 0.3
}

# Add info about API billing
st.sidebar.info("Note: Using these APIs will incur charges to your account based on the selected model and usage. Ollama is self-hosted and free to use.")

# Initialize clients with user-provided keys
openai_client = None
anthropic_client = None

if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
    
if anthropic_api_key:
    anthropic_client = Anthropic(api_key=anthropic_api_key)

# Model selection
llm_provider = st.radio("Select LLM Provider:", ["OpenAI", "Anthropic (Claude)", "Ollama (Self-hosted)"])

if llm_provider == "OpenAI":
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    selected_model = st.selectbox("Select OpenAI Model:", model_options, index=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=temperature_values["OpenAI"], step=0.1,
                           help="Lower values make responses more focused and deterministic. Higher values make output more random and creative.")
elif llm_provider == "Anthropic (Claude)":
    model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
    selected_model = st.selectbox("Select Claude Model:", model_options, index=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=temperature_values["Anthropic (Claude)"], step=0.1,
                           help="Lower values make responses more focused and deterministic. Higher values make output more random and creative.")
else:  # Ollama
    # These are common Ollama models, but users might have others available
    model_options = ["llama3.3:70b-instruct-q8_0", "qwen3:32b-q8_0", "phi4-reasoning:14b-plus-fp16"]
    selected_model = st.selectbox("Select Ollama Model:", model_options, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=temperature_values["Ollama (Self-hosted)"], step=0.1,
                           help="Lower values make responses more focused and deterministic. Higher values make output more random and creative.")

# Add option for number of self-correction attempts
st.sidebar.markdown("**Number of Attempts to Optimize the RDF/SHACL files**")
max_attempts_opimization = st.sidebar.number_input(
    "How many attempt to generate RDF/SHACL data?", 
    min_value=1, 
    max_value=10, 
    value=3,
    help="Number of times the LLM should attempt to generate RDF/SHACL"
)

# Add option for number of self-correction attempts
st.sidebar.markdown("**Self-Correction Attempts for Validation Part**")
max_attempts_correction = st.sidebar.number_input(
    "How many attempt to correct RDF/SHACL data to pass the validation process?", 
    min_value=1, 
    max_value=10, 
    value=3,
    help="Number of times the LLM should attempt to fix RDF/SHACL after validation fails"
)

# User input
# Example data option
st.subheader("Input Options")
use_example = st.checkbox("Use example data")

creep_test_example = """BAM 5.2 Vh5205_C-95.LIS						
------------------------------------						
ENTRY	SYMBOL	UNIT		* Information common to all tests		
Date of test start			30.8.23 9:06 AM			
Test ID			Vh5205_C-95			
Test standard			DIN EN ISO 204:2019-4	*		
Specified temperature	T	?	980 °C	*		
Type of loading			Tension	*		
Initial stress	Ro	MPa	140			
(Digital) Material Identifier			CMSX-6	*		
"Description of the manufacturing process - as-tested material
"			Single Crystal Investment Casting from a Vacuum Induction Refined Ingot and subsequent Heat Treatment (annealed and aged).	*		
Single crystal orientation		°	7,5			
Type of test piece II			Round cross section	*		
Type of test piece III			Smooth test piece	*		
Sensor type - Contacting extensometer			Clip-on extensometer	*		
Min. test piece diameter at room temperature	D	mm	5,99			
Reference length for calculation of percentage elongations	Lr = Lo	mm	23,9			
Reference length for calculation of percentage extensions	Lr = Le	mm	22,9			
Heating time		h	1,61			
Soak time before the test		h	2,81			
Test duration	t	h	1010			
Creep rupture time	tu	h	Not applicable			
Percentage permanent elongation	Aper	%	1,14			
Percentage elongation after creep fracture	Au	%	Not applicable			
Percentage reduction of area after creep fracture	Zu	%	Not applicable			
Percentage total extension	et	%	0,964			
Percentage initial total extension	eti	%	0,153			
Percentage elastic extension	ee	%	0,153			
Percentage initial plastic extension	ei	%	0			
Percentage plastic extension	ep	%	0,811			
Percentage creep extension	ef	%	0,811"""

# Modify the user input text area to show the example if checked
if use_example:
    data_input = st.text_area("Mechanical test data:", value=creep_test_example, height=300)
else:
    data_input = st.text_area("Paste your mechanical test data (e.g. JSON or description):", height=200)
generate = st.button("Generate RDF & SHACL")

# Function to call Ollama API
def call_ollama_api(endpoint, model, prompt, system_prompt=None, temperature=0.7):
    headers = {
        "Content-Type": "application/json"
    }
    
    # Format the request based on whether a system prompt is provided
    if system_prompt:
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": temperature
        }
    else:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
    
    try:
        # Use the chat endpoint
        chat_url = f"{endpoint.rstrip('/')}/api/chat"
        response = requests.post(chat_url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            # Try the generate endpoint as fallback for older Ollama versions
            generate_url = f"{endpoint.rstrip('/')}/api/generate"
            generate_data = {"model": model, "prompt": prompt, "temperature": temperature}
            response = requests.post(generate_url, json=generate_data, headers=headers)
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                st.error(f"Ollama API Error: {response.status_code} - {response.text}")
                return f"Error: Unable to get response from Ollama API. Status code: {response.status_code}"
    except Exception as e:
        st.error(f"Error connecting to Ollama API: {str(e)}")
        return f"Error connecting to Ollama API: {str(e)}"

def extract_rdf_shacl(response_text):
    parts = response_text.split("```")
    rdf_code = ""
    shacl_code = ""

    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a code block
            if "turtle" in part.lower() or "@prefix" in part:
                part_clean = part.replace("turtle", "").strip()
                if not rdf_code:
                    rdf_code = part_clean
                elif not shacl_code:
                    shacl_code = part_clean
    return rdf_code, shacl_code


def extract_rdf_shacl_improved(response_text):
    """Improved extraction with better error handling and debugging"""
    if not response_text or not response_text.strip():
        st.error("❌ Empty response from LLM")
        return "", ""
    
    # Log the response for debugging
    # with st.expander("🔍 Debug: Raw LLM Response"):
    #     st.text(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
    
    parts = response_text.split("```")
    rdf_code = ""
    shacl_code = ""
    
    # Find turtle code blocks
    code_blocks = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a code block
            part_clean = part.strip()
            # Remove language identifier if present
            if part_clean.startswith("turtle"):
                part_clean = part_clean[6:].strip()
            elif part_clean.startswith("ttl"):
                part_clean = part_clean[3:].strip()
            
            # Check if it looks like RDF/SHACL (contains @prefix or sh:)
            if "@prefix" in part_clean or "sh:" in part_clean:
                code_blocks.append(part_clean)
    
    # Assign first block to RDF, second to SHACL
    if len(code_blocks) >= 1:
        rdf_code = code_blocks[0]
    if len(code_blocks) >= 2:
        shacl_code = code_blocks[1]
    else:
        # If only one block, try to split by detecting SHACL patterns
        if rdf_code and ("sh:" in rdf_code or "Shape" in rdf_code):
            lines = rdf_code.split('\n')
            rdf_lines = []
            shacl_lines = []
            in_shacl = False
            
            for line in lines:
                if "sh:" in line or "Shape" in line:
                    in_shacl = True
                if in_shacl:
                    shacl_lines.append(line)
                else:
                    rdf_lines.append(line)
            
            if shacl_lines:
                rdf_code = '\n'.join(rdf_lines)
                shacl_code = '\n'.join(shacl_lines)
    
    # st.info(f"📊 Extracted RDF: {len(rdf_code)} chars, SHACL: {len(shacl_code)} chars")
    return rdf_code, shacl_code

def generate_rdf_shacl_with_debugging(prompt_text, system_prompt, model_info, previous_output=None):
    """Enhanced generation with better error handling and debugging"""
    
    improvement_instruction = ""
    if previous_output:
        improvement_instruction = (
            "\n\nIMPROVE THE FOLLOWING RDF AND SHACL:\n\n"
            f"{previous_output}"
        )
        prompt_text = f"{prompt_text}{improvement_instruction}"
    
    # Log prompt length for debugging
    # st.info(f"📏 Prompt length: {len(prompt_text)} characters")
    
    try:
        if model_info['provider'] == "OpenAI":
            if not openai_client:
                st.error("❌ OpenAI client not initialized")
                return "", ""
                
            response = openai_client.chat.completions.create(
                model=model_info['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=model_info['temperature'],
                max_tokens=4000
            )
            content = response.choices[0].message.content
            
        elif model_info['provider'] == "Anthropic (Claude)":
            if not anthropic_client:
                st.error("❌ Anthropic client not initialized")
                return "", ""
                
            response = anthropic_client.messages.create(
                model=model_info['model'],
                max_tokens=4000,
                temperature=model_info['temperature'],
                system=system_prompt,
                messages=[{"role": "user", "content": prompt_text}]
            )
            content = response.content[0].text
            
        else:  # Ollama
            content = call_ollama_api(
                model_info['endpoint'],
                model_info['model'],
                prompt_text,
                system_prompt,
                temperature=model_info['temperature']
            )
        
        if not content or not content.strip():
            st.error("❌ Empty response from LLM")
            return "", ""
            
        return extract_rdf_shacl_improved(content)
        
    except Exception as e:
        st.error(f"❌ Error calling LLM: {str(e)}")
        return "", ""


def generate_rdf_shacl(prompt_text, system_prompt, model_info, previous_output=None):
    improvement_instruction = ""
    if previous_output:
        improvement_instruction = (
            "\n\nYour task is to improve the following RDF and SHACL for structure, correctness, "
            "ontology mapping, and completeness:\n\n"
            f"{previous_output}"
        )
        prompt_text = f"{prompt_text}{improvement_instruction}"

    if model_info['provider'] == "OpenAI":
        response = openai_client.chat.completions.create(
            model=model_info['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=model_info['temperature'],
        )
        content = response.choices[0].message.content
    elif model_info['provider'] == "Anthropic (Claude)":
        response = anthropic_client.messages.create(
            model=model_info['model'],
            max_tokens=4000,
            temperature=model_info['temperature'],
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_text}]
        )
        content = response.content[0].text
    else:  # Ollama
        content = call_ollama_api(
            model_info['endpoint'],
            model_info['model'],
            prompt_text,
            system_prompt,
            temperature=model_info['temperature']
        )

    return extract_rdf_shacl(content)


def validate_rdf_shacl(rdf_code, shacl_code):

    try:
        # Parse RDF with better error handling
        rdf_graph = Graph()
        rdf_graph.parse(data=rdf_code, format="turtle")
        
        # Parse SHACL with better error handling  
        shacl_graph = Graph()
        shacl_graph.parse(data=shacl_code, format="turtle")

        # Validate with more detailed reporting
        conforms, results_graph, results_text = validate(
            data_graph=rdf_graph,
            shacl_graph=shacl_graph,
            inference='rdfs',
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
            debug=False
        )
        
        # Return more detailed results
        if results_text:
            return conforms, results_text
        else:
            return conforms, "Validation completed successfully" if conforms else "Validation failed with unknown errors"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}\n\nThis might be due to invalid Turtle syntax. Please check the RDF and SHACL code for syntax errors."



def get_correction_explanation(rdf_code, shacl_code, model_info):
    critique_prompt = f"""
You are a senior materials science and knowledge graph engineer with expertise in semantic web technologies and ontology.

Please perform a detailed technical critique of the following RDF and SHACL output for a materials science knowledge graph and ontology:

RDF:
{rdf_code}

SHACL:
{shacl_code}

Provide a structured analysis covering:

1. SEMANTIC COHERENCE
   - Are domain concepts accurately represented?
   - Are relationships between entities semantically valid?
   - Is there proper use of materials science terminology?

2. STRUCTURAL INTEGRITY
   - Evaluate triple patterns and graph structure
   - Identify any disconnected nodes or subgraphs
   - Assess consistency in URI/IRI patterns

3. ONTOLOGY ALIGNMENT
   - Is the schema aligned with standard materials science ontologies (e.g., EMMO, MatOnto, ChEBI)?
   - Are there opportunities to link to established domain vocabularies?
   - Suggest specific namespace improvements

4. COMPLETENESS
   - Identify missing critical properties for materials characterization
   - Assess coverage of key materials relationships (composition, structure, properties, processing)
   - Evaluate sufficiency of metadata (provenance, units, measurement conditions)

5. SHACL VALIDATION
   - Are constraints appropriate for the domain?
   - Are validation rules comprehensive?
   - Are there missing constraints for ensuring data quality?

6. BEST PRACTICES
   - Conformance to Linked Open Data principles
   - Proper use of rdf:type, rdfs:subClassOf, owl:equivalentClass, etc.
   - Appropriate use of literals vs. URIs

Format your response with bullet points organized by these categories, and conclude with 2-3 highest priority recommendations for improvement.
"""

    if model_info['provider'] == "OpenAI":
        response = openai_client.chat.completions.create(
            model=model_info['model'],
            messages=[
                {"role": "system", "content": "You are a knowledge graph and ontology critique assistant."},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif model_info['provider'] == "Anthropic (Claude)":
        response = anthropic_client.messages.create(
            model=model_info['model'],
            max_tokens=3000,
            temperature=temperature,
            system="You are a knowledge graph and ontology critique assistant.",
            messages=[
                {"role": "user", "content": critique_prompt}
            ]
        )
        return response.content[0].text

    else:  # Ollama
        return call_ollama_api(
            model_info['endpoint'],
            model_info['model'],
            critique_prompt,
            system_prompt="You are a knowledge graph and ontology critique assistant.",
            temperature=temperature
        )


# Update the generate button check
if generate and data_input.strip():
    # Check if API keys are provided based on selected model
    if llm_provider == "OpenAI" and not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif llm_provider == "Anthropic (Claude)" and not anthropic_api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif llm_provider == "Ollama (Self-hosted)" and not ollama_endpoint:
        st.error("Please enter your Ollama API endpoint in the sidebar.")
    else:
        # with st.spinner(f"Calling {llm_provider} to generate RDF and SHACL..."):
        with st.spinner("Generating RDF & SHACL with self-correction..."):
            model_info = {
                "provider": llm_provider,
                "model": selected_model,
                "temperature": temperature,
                "endpoint": ollama_endpoint
            }
            system_prompt = """
# Materials Science Knowledge Graph Expert

You are a specialized knowledge engineer for materials science, focusing on transforming unstructured creep test reports into standardized RDF and SHACL models. Your expertise bridges materials science domain knowledge with semantic web technologies.

## Core Competencies
- Converting materials testing data into formal ontology structures
- Creating valid, interoperable RDF representations of experimental data
- Generating SHACL shapes for validation and model conformance
- Maintaining knowledge graph best practices in materials science domains

## Structured Reasoning Approach

For each transformation task, follow this refined methodology:

### 1. Extract Entities and Concepts
- **Materials**: Composition, processing history, classification
- **Test Equipment**: Instruments, calibration status, standards compliance
- **Test Parameters**: Temperature, stress, atmosphere, loading protocols
- **Measurements**: Time series data, strain values, derived calculations
- **Personnel**: Operators, supervisors, analysts
- **Documentation**: Standards, procedures, certifications

### 2. Ontological Mapping
- Map each entity to appropriate ontology classes using:
  - Materials Workflow (`matwerk:`) - For material samples and testing procedures
  - Ontology for Biomedical Investigations (`obi:`) - For experimental processes
  - Information Artifact Ontology (`iao:`) - For documentation elements
  - NFDI/PMD Core (`nfdi:`, `pmd:`) - For domain-specific concepts
  - QUDT (`qudt:`, `unit:`) - For quantities, units, dimensions

### 3. Define Semantic Relationships
- Create object property networks reflecting physical and conceptual connections
- Establish provenance chains for data traceability using:
  - `obi:has_specified_input`/`obi:has_specified_output`
  - `prov:wasGeneratedBy`/`prov:wasDerivedFrom`
  - `matwerk:hasProperty`/`matwerk:hasFeature`
- Include bidirectional relationships with inverse properties

### 4. Quantitative Data Modeling
- Model all numerical values using the QUDT pattern:
  - Create quantity instances (e.g., `ex:temperature_value_001`)
  - Attach numerical values with `qudt:numericValue` and proper XSD types
  - Specify measurement units via `qudt:unit`
  - Include `qudt:standardUncertainty` where available

### 5. Temporal Data Representation
- Create observation collections with `time:Instant` timestamps
- Link time series data points to the relevant phase of creep behavior
- Maintain interval relationships for capturing test sequence

### 6. IRI Engineering & Metadata Enhancement
- Generate consistent, hierarchical IRIs following materials science conventions
- Add `rdfs:label` and `rdfs:comment` to ALL resources
- Include contextual metadata like creation date, version, and provenance

### 7. RDF Generation (Turtle Format)
- Create a complete, valid RDF document with comprehensive prefix declarations
- Group related triples for readability
- Include all required metadata and contextual information
- Follow W3C best practices for RDF representation

### 8. SHACL Shape Development
- Create node shapes for all major entity types
- Define property shapes with cardinality, value types, and constraints
- Include `sh:description` for human-readable validation messages
- Enforce required properties and data consistency rules

### 9. Validation & Refinement
- Test RDF against SHACL constraints
- Diagnose and resolve any validation issues
- Optimize for data quality and semantic correctness

## Required Namespace Declarations

Both RDF and SHACL outputs must include all of these prefixes:

```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/ns#> .
@prefix matwerk: <http://matwerk.org/ontology#> .
@prefix nfdi: <http://nfdi.org/ontology/core#> .
@prefix pmd: <http://pmd.org/ontology/core#> .
@prefix iao: <http://purl.obolibrary.org/obo/IAO_> .
@prefix obi: <http://purl.obolibrary.org/obo/OBI_> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
```

## Required Entity Types and Properties

### Core Material Entities
- `matwerk:MaterialSample` - Physical specimen undergoing testing
- `matwerk:Material` - Composition and classification of material
- `matwerk:MaterialProperty` - Properties like strength, ductility

### Experimental Process Entities
- `matwerk:CreepTest` - Main testing process
- `obi:assay` - General experimental process
- `obi:material_processing` - Sample preparation steps

### Information Entities
- `iao:document` - Test reports, procedures, standards
- `iao:measurement_datum` - Raw measurements
- `iao:data_set` - Collections of related measurements

### Measurement Entities
- `qudt:Quantity` - Quantitative values with units
- `time:Instant` - Temporal reference points
- `time:Interval` - Test duration periods

## Detailed Data Modeling Requirements

### 1. Sample Metadata Requirements
- Sample identification with traceable IRI pattern
- Material composition (elements and percentages)
- Processing history and heat treatment details
- Physical dimensions (gauge length, cross-section)
- Microstructural characteristics when available

### 2. Test Configuration Requirements
- Testing standard compliance (ASTM, ISO, etc.)
- Equipment details with calibration status
- Specimen geometry and orientation
- Environmental parameters (temperature, atmosphere)
- Loading conditions and control parameters

### 3. Measurement Requirements
- Time-strain data series with timestamps
- Creep rate calculations for each stage
- Rupture time or test termination point
- Derived properties (minimum creep rate, strain at rupture)
- Measurement uncertainties and confidence intervals

### 4. Results Representation Requirements
- Structured representation of primary, secondary, and tertiary creep phases
- Statistical summaries of key parameters
- Links to raw data and derived calculations
- Observations and analysis notes

## Output Deliverables

For each creep test report, generate two distinct artifacts:

1. **Complete RDF Data Model (Turtle format)**
   - Comprehensive representation of all extracted information
   - Properly typed entities with descriptive labels
   - Complete relationship network
   - Valid syntax with all required prefixes

2. **SHACL Validation Shape (Turtle format)**
   - Shape constraints matching exactly the RDF data structure
   - Property constraints with appropriate cardinality
   - Data type and value range enforcement
   - Validation reporting capabilities

Both outputs must be syntactically valid and semantically aligned with materials science domain knowledge. The SHACL shape must successfully validate the RDF data, producing a conformant validation report.

## Implementation Guidelines

- Use ontology design patterns from OBO Foundry when applicable
- Apply consistent naming conventions for all resources
- Include human-readable labels and descriptions for all entities
- Structure data hierarchically for navigation and query efficiency
- Ensure all numerical values have appropriate XSD datatypes
- Validate RDF and SHACL for syntax correctness before submission
"""
            ### 1️⃣ INITIAL GENERATION
            # with st.spinner("🔄 Generating initial RDF & SHACL..."):
            # Generate initial RDF & SHACL
            rdf_code, shacl_code = generate_rdf_shacl(data_input, system_prompt, model_info)
            st.subheader("🟢 Initial RDF Output")
            st.code(rdf_code, language="turtle")
            st.subheader("🟢 Initial SHACL Output")
            st.code(shacl_code, language="turtle")

            # Run correction iterations BEFORE validation based on user-defined attempts
            for i in range(1, max_attempts_opimization):  # Already did 1 above, so run up to max_attempts-1 more
                st.markdown(f"### 🔄 Optimization Pass {i}")
                explanation = get_correction_explanation(rdf_code, shacl_code, model_info)
                st.markdown(f"### 🧠 Why Improve This (Pass {i})")
                st.info(explanation)

                with st.spinner(f"🔄 Optimizing RDF/SHACL (Step {i})..."):
                    rdf_code, shacl_code = generate_rdf_shacl(
                        data_input,
                        system_prompt,
                        model_info,
                        previous_output=f"{rdf_code}\n\n{shacl_code}\n\nCritique:\n{explanation}"
                    )
                
                st.subheader(f"🟡 Optimized RDF v{i}")
                st.code(rdf_code, language="turtle")
                st.subheader(f"🟡 Optimized SHACL v{i}")
                st.code(shacl_code, language="turtle")

            if not rdf_code.strip() or not shacl_code.strip():
                st.error("RDF or SHACL content is empty after pre-validation optimization.")
                st.stop()

            # --- SHACL validation and retry correction loop ---
            st.subheader("✅ SHACL Validation")
            conforms, validation_report = validate_rdf_shacl(rdf_code, shacl_code)

            # max_attempts = 1
            # max_attempts is now set by the user from the sidebar
            attempt = 0

            if conforms:
                st.success("✅ SHACL Validation: PASSED")
                with st.expander("📋 Validation Report"):
                    st.code(validation_report)

                # 🟢 DISPLAY FINAL VALIDATED RDF & SHACL
                st.subheader("🎯 Final Validated RDF & SHACL")
                st.markdown("**The following RDF and SHACL have passed validation and are ready for use:**")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📄 **Final RDF Data**")
                    st.code(rdf_code, language="turtle")
                    st.download_button(
                        label="⬇️ Download Final RDF",
                        data=rdf_code,
                        file_name="validated_mechanical_test.ttl",
                        mime="text/turtle",
                        key="download_validated_rdf"
                    )
                with col2:
                    st.markdown("### 🛡️ **Final SHACL Shapes**")
                    st.code(shacl_code, language="turtle")
                    st.download_button(
                        label="⬇️ Download Final SHACL",
                        data=shacl_code,
                        file_name="validated_mechanical_test_shapes.ttl",
                        mime="text/turtle",
                        key="download_validated_shacl"
                    )

                # Model summary
                # st.markdown("### 📊 **Model Summary**")
                # try:
                #     g = Graph()
                #     g.parse(data=rdf_code, format="turtle")
                #     st.metric("Total Triples", len(g))
                #     st.metric("Unique Subjects", len(set(g.subjects())))
                #     st.metric("Unique Predicates", len(set(g.predicates())))
                #     st.metric("Unique Objects", len(set(g.objects())))
                # except Exception as e:
                #     st.info("Could not generate model statistics")

            else:

                while not conforms and attempt < max_attempts_correction:
                    attempt += 1
                    st.warning(f"❌ Validation failed. Attempting fix #{attempt}/{max_attempts_correction}...")
                    
                    # Show the current validation errors for debugging
                    with st.expander(f"🔍 Validation Errors (Attempt {attempt})"):
                        st.code(validation_report)
                    
                    # Simplified and focused retry prompt
                    retry_prompt = f"""Fix the SHACL validation errors in the following RDF and SHACL data.

                VALIDATION ERRORS:
                {validation_report}

                INSTRUCTIONS:
                1. Fix all validation errors listed above
                2. Return corrected RDF in first ```turtle block
                3. Return corrected SHACL in second ```turtle block
                4. Keep all namespace prefixes
                5. Ensure RDF conforms to SHACL shapes

                CURRENT RDF:
                ```turtle
                {rdf_code}
                ```

                CURRENT SHACL:
                ```turtle
                {shacl_code}
                ```

                Return the corrected versions now:"""

                    # Generate corrected versions with debugging
                    try:
                        with st.spinner(f"🔧 Generating corrections (attempt {attempt})..."):
                            new_rdf, new_shacl = generate_rdf_shacl_with_debugging(
                                retry_prompt, 
                                "You are an expert at fixing RDF and SHACL validation errors.", 
                                model_info
                            )
                        
                        # Validate the extracted content
                        if not new_rdf.strip():
                            st.error(f"❌ No RDF extracted from LLM response (attempt {attempt})")
                            continue
                            
                        if not new_shacl.strip():
                            st.error(f"❌ No SHACL extracted from LLM response (attempt {attempt})")
                            continue
                        
                        # Update the working versions
                        rdf_code = new_rdf
                        shacl_code = new_shacl
                        
                        # Show what was generated for this attempt
                        # with st.expander(f"🔧 Generated Fix (Attempt {attempt})"):
                        #     st.markdown("**RDF Preview:**")
                        #     st.code(rdf_code[:300] + "..." if len(rdf_code) > 300 else rdf_code, language="turtle")
                        #     st.markdown("**SHACL Preview:**")
                        #     st.code(shacl_code[:300] + "..." if len(shacl_code) > 300 else shacl_code, language="turtle")
                        
                        # Validate the corrected versions
                        st.info(f"🔍 Validating corrected version (attempt {attempt})...")
                        conforms, validation_report = validate_rdf_shacl(rdf_code, shacl_code)
                        
                                    # Display final validation result
                        if conforms:
                            st.success("✅ Final SHACL Validation: PASSED")
                            with st.expander("📋 Validation Report"):
                                st.code(validation_report)
                            
                            # ---------- DISPLAY FINAL VALIDATED RDF & SHACL ----------
                            st.subheader("🎯 Final Validated RDF & SHACL")
                            st.markdown("**The following RDF and SHACL have passed validation and are ready for use:**")
                            
                            # Create two columns for side-by-side display
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 📄 **Final RDF Data**")
                                st.code(rdf_code, language="turtle")
                                
                                # Download button for RDF
                                st.download_button(
                                    label="⬇️ Download Final RDF",
                                    data=rdf_code,
                                    file_name="validated_mechanical_test.ttl",
                                    mime="text/turtle",
                                    help="Download the validated RDF data in Turtle format"
                                )
                            
                            with col2:
                                st.markdown("### 🛡️ **Final SHACL Shapes**")
                                st.code(shacl_code, language="turtle")
                                
                                # Download button for SHACL
                                st.download_button(
                                    label="⬇️ Download Final SHACL",
                                    data=shacl_code,
                                    file_name="validated_mechanical_test_shapes.ttl",
                                    mime="text/turtle",
                                    help="Download the validated SHACL shapes in Turtle format"
                                )
                            
            #                 # Optional: Add a combined download option
            #                 st.markdown("### 📦 **Combined Download**")
            #                 combined_content = f"""# Validated RDF Data
            # # Generated on: {st.session_state.get('timestamp', 'Unknown')}
            # # Validation Status: PASSED

            # {rdf_code}

            # # ==========================================
            # # SHACL Shapes for Validation
            # # ==========================================

            # {shacl_code}
            # """
                            
            #                 st.download_button(
            #                     label="⬇️ Download Both RDF + SHACL (Combined)",
            #                     data=combined_content,
            #                     file_name="validated_complete_model.ttl",
            #                     mime="text/turtle",
            #                     help="Download both RDF and SHACL in a single file"
            #                 )
                            
                            # Add summary statistics
                            st.markdown("### 📊 **Model Summary**")
                            try:
                                # Parse RDF to get some basic statistics
                                temp_graph = Graph()
                                temp_graph.parse(data=rdf_code, format="turtle")
                                
                                # Count triples, subjects, predicates, objects
                                num_triples = len(temp_graph)
                                subjects = set(temp_graph.subjects())
                                predicates = set(temp_graph.predicates())
                                objects = set(temp_graph.objects())
                                
                                # Display statistics in columns
                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                
                                with stat_col1:
                                    st.metric("Total Triples", num_triples)
                                with stat_col2:
                                    st.metric("Unique Subjects", len(subjects))
                                with stat_col3:
                                    st.metric("Unique Predicates", len(predicates))
                                with stat_col4:
                                    st.metric("Unique Objects", len(objects))
                                
                            except Exception as e:
                                st.info("Could not generate model statistics")
                            
                        else:
                            st.error("❌ Final SHACL Validation: FAILED")
                            with st.expander("📋 Final Validation Report"):
                                st.code(validation_report)
                            st.info("💡 Tip: The RDF and SHACL may still be usable despite validation issues. Check the specific errors above.")
                            
                            # Even if validation fails, still show the data with download options
                            st.subheader("⚠️ RDF & SHACL (Validation Failed after {attempts} attempts)")
                            st.markdown("**Note: The following data failed validation but may still be useful:**")
                            
                            # Create two columns for side-by-side display
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 📄 **RDF Data (Unvalidated)**")
                                st.code(rdf_code, language="turtle")
                                
                                # Download button for RDF
                                st.download_button(
                                    label="⬇️ Download RDF (Unvalidated)",
                                    data=rdf_code,
                                    file_name="unvalidated_mechanical_test.ttl",
                                    mime="text/turtle",
                                    help="Download the RDF data (validation failed)",
                                    key="download_unvalidated_rdf_{attempt}"  # Unique key for each attempt to avoid conflicts
                                )
                            
                            with col2:
                                st.markdown("### 🛡️ **SHACL Shapes (Unvalidated)**")
                                st.code(shacl_code, language="turtle")
                                
                                # Download button for SHACL
                                st.download_button(
                                    label="⬇️ Download SHACL (Unvalidated)",
                                    data=shacl_code,
                                    file_name="unvalidated_mechanical_test_shapes.ttl",
                                    mime="text/turtle",
                                    help="Download the SHACL shapes (validation failed)",
                                    key="download_unvalidated_shacl_{attempt}"  # Unique key for each attempt to avoid conflicts
                                )
                            
                    except Exception as e:
                        st.error(f"❌ Error during correction attempt {attempt}: {str(e)}")
                        break


        # ---------- Ontology Term Suggestion ----------
        st.subheader("🔎 Suggested Ontology Terms")

        term_prompt = f"""
        You are an ontology expert specializing in material science. 
        Your task is to map each field listed below to an appropriate ontology class or property 
        from one of the following sources: EMMO, MATWERK, PMDcore, NFDI Core, IAO, OBI, OBO, or QUDT.

        Instructions:
        - Only suggest classes or properties that truly exist in the specified ontologies.
        - If no suitable match is found, clearly respond with "No match found" instead of guessing.
        - For each field, provide:
            - Ontology name
            - Class or property name
            - (Optional) Ontology ID or link, if available.

        Input fields:
        {data_input}
        """

        # Get ontology suggestions using the selected model
        if llm_provider == "OpenAI":
            ontology_response = openai_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You're an ontology assistant for material science."},
                    {"role": "user", "content": term_prompt}
                ],
                temperature=temperature,
            )
            ontology_content = ontology_response.choices[0].message.content
        elif llm_provider == "Anthropic (Claude)":
            ontology_response = anthropic_client.messages.create(
                model=selected_model,
                max_tokens=6000,
                temperature=temperature,
                system="You're an ontology assistant for material science.",
                messages=[
                    {"role": "user", "content": term_prompt}
                ]
            )
            ontology_content = ontology_response.content[0].text
        else:  # Ollama
            system_prompt_ontology = "You're an ontology assistant for material science."
            ontology_content = call_ollama_api(ollama_endpoint, selected_model, term_prompt, system_prompt_ontology, temperature)

        st.markdown(ontology_content)

        # ---------- Visualize with NetworkX + Pyvis ----------
        st.subheader("🌐 RDF Graph Visualization")

        # Update the visualization section with larger dimensions and improved layout settings
        def visualize_rdf(rdf_text):
            g = Graph().parse(data=rdf_text, format="turtle")
            nx_graph = nx.DiGraph()

            for s, p, o in g:
                nx_graph.add_edge(str(s), str(o), label=str(p))

            # Create a larger network with improved physics settings
            net = Network(height="900px", width="100%", directed=True, notebook=False)
            
            # Configure physics for better graph spacing
            net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
            
            # Increase node spacing
            net.repulsion(node_distance=300, central_gravity=0.01, spring_length=300, spring_strength=0.05, damping=0.09)
            
            # Add nodes with larger size
            # Inside your visualize_rdf function, update the node addition logic:
            for node in nx_graph.nodes:
                # Extract shorter node labels for readability
                short_label = node.split("/")[-1] if "/" in node else node
                short_label = short_label.split("#")[-1] if "#" in short_label else short_label
                
                # Check if this is a blank node (starts with 'n' followed by numbers)
                is_blank_node = bool(re.match(r'^n\d+$', short_label))
                
                # Use different styling for blank nodes
                if is_blank_node:
                    node_color = "#E8E8E8"  # Light gray
                    node_size = 15  # Smaller size
                    label = ""  # Hide the label
                else:
                    node_color = "#97C2FC"  # Default blue
                    node_size = 25  # Normal size
                    label = short_label
                
                net.add_node(node, label=label, size=node_size, 
                            color=node_color, font={'size': 16}, 
                            title=node)  # Title shows on hover

            # Add edges with better visibility
            for u, v, d in nx_graph.edges(data=True):
                # Extract shorter edge labels
                edge_label = d["label"].split("/")[-1] if "/" in d["label"] else d["label"]
                edge_label = edge_label.split("#")[-1] if "#" in edge_label else edge_label
                
                net.add_edge(u, v, label=edge_label, font={'size': 12}, width=1.5, title=d["label"])

            # Set options for better visualization
            net.set_options("""
            const options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "iterations": 100,
                        "updateInterval": 10,
                        "fit": true
                    },
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "springLength": 250,
                        "springConstant": 0.04,
                        "damping": 0.09
                    }
                },
                "layout": {
                    "improvedLayout": true,
                    "hierarchical": {
                        "enabled": false
                    }
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hover": true,
                    "multiselect": true,
                    "tooltipDelay": 100
                }
            }
            """)

            tmp_dir = tempfile.mkdtemp()
            html_path = os.path.join(tmp_dir, f"graph_{uuid.uuid4()}.html")
            net.save_graph(html_path)
            return html_path

        # Update the HTML component size
        html_file = visualize_rdf(rdf_code)
        with open(html_file, 'r', encoding='utf-8') as f:
            graph_html = f.read()

        # Render with larger dimensions
        components.html(graph_html, height=1000, width=1200, scrolling=True)

        # Add instructions for graph interaction
        st.markdown("""
        ### Graph Navigation Instructions:
        - **Zoom**: Use mouse wheel or pinch gesture
        - **Pan**: Click and drag empty space
        - **Move nodes**: Click and drag nodes to rearrange
        - **View details**: Hover over nodes or edges for full information
        - **Select multiple**: Hold Ctrl or Cmd while clicking nodes
        - **Reset view**: Double-click on empty space
        """)