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
    model_options = ["llama3.3:latest", "deepseek-r1:70b", "gemma3:27b", "llama3.2-vision:90b-instruct-q8_0"]
    selected_model = st.selectbox("Select Ollama Model:", model_options, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=temperature_values["Ollama (Self-hosted)"], step=0.1,
                           help="Lower values make responses more focused and deterministic. Higher values make output more random and creative.")

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
        with st.spinner(f"Calling {llm_provider} to generate RDF and SHACL..."):
            system_prompt = """
You are an expert in semantic data modeling for materials science. Your task is to transform mechanical creep test reports into structured RDF and SHACL models.

When presented with a creep test report, generate:
1. Complete RDF data in Turtle (.ttl) format
2. Corresponding SHACL shape in Turtle (.ttl) format for validation

**Mandatory namespaces to declare explicitly in both RDF and SHACL:**
- ex: <http://example.org/ns#>
- xsd: <http://www.w3.org/2001/XMLSchema#>
- rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
- rdfs: <http://www.w3.org/2000/01/rdf-schema#>
- owl: <http://www.w3.org/2002/07/owl#>
- matwerk: <http://matwerk.org/ontology#>
- nfdi: <http://nfdi.org/ontology/core#>
- pmd: <http://pmd.org/ontology/core#>
- iao: <http://purl.obolibrary.org/obo/IAO_>
- obi: <http://purl.obolibrary.org/obo/OBI_>
- obo: <http://purl.obolibrary.org/obo/>
- qudt: <http://qudt.org/schema/qudt/>
- unit: <http://qudt.org/vocab/unit/>
- time: <http://www.w3.org/2006/time#>

**Ontological Structure Requirements:**
1. **Classes**:
   - Define necessary classes with proper hierarchies.
   - Use `matwerk:CreepTest`, `matwerk:MaterialSample`, and related terms.
   - Apply `OBI` experimental process classes where appropriate.
   - Include `IAO` information entity classes for documentation.
   - Create measurement classes aligned with `NFDI/PMD` core ontologies.

2. **Object Properties**:
   - Define relationships between entities.
   - Set domain and range constraints.
   - Include inverse properties where applicable.
   - Follow `OBO Relation Ontology` patterns.
   - Link test processes to their corresponding inputs and outputs.

3. **DataType Properties**:
   - Create properties for scalar measurements (e.g., temperatures, forces).
   - Use appropriate `xsd` datatypes (`xsd:float`, `xsd:date`, etc.).
   - Include value constraints where applicable.
   - Attach measurement units using `qudt:unit`.

4. **Individuals**:
   - Create properly typed instances.
   - Assign unique IRIs consistently (e.g., `ex:sample_{id}`).
   - Connect individuals to related class instances.
   - Include measurement values with correct xsd datatypes.

**Modeling Requirements:**

1. **Sample Metadata**:
   - Assign a unique sample IRI (e.g., `ex:sample_001`).
   - Specify material composition and classification (`matwerk:hasMaterial`).
   - Record the test date (`xsd:date`).
   - Link the test to a project (`iao:part_of` some project entity).
   - Include operator details and their role (`obi:operator_role`).

2. **Test Configuration**:
   - Record testing standard used (e.g., ASTM E139-11).
   - Document equipment details (model, calibration date).
   - Specify specimen geometry (shape, gauge length, diameter/thickness).
   - Mention material orientation relative to production processes.
   - Record environmental conditions (e.g., atmosphere, humidity).

3. **Test Parameters**:
   - Applied force (convert and record in Newtons).
   - Test temperature (record in Kelvin).
   - Applied stress (record in MPa).
   - Target strain rates and loading protocols.
   - Hold time duration (if applicable).

4. **Results Representation**:
   - Create distinct IRIs for each measurement (e.g., initial strain, creep rate).
   - Capture primary, secondary, and tertiary creep stages.
   - Record final strain values and rupture times if applicable.
   - Link results to the relevant test and sample.

5. **Time Series Modeling**:
   - Create an observation collection for time-strain data.
   - Each observation must record a timestamp (hours) and corresponding strain (%).
   - Include measurement uncertainties where available.
   - Properly link observations to associated test parameters.

**RDF and SHACL Implementation Rules:**

- Use descriptive predicates from standard ontologies whenever possible.
- Apply the `qudt:Quantity` modeling pattern for all measurements with associated units.
- Build class hierarchies consistent with `OBO Foundry` principles.
- Ensure `rdfs:label` and `rdfs:comment` are added for human readability of every resource.
- Always format numerical values precisely using correct `xsd` datatypes (e.g., `xsd:float`, `xsd:integer`).
- Use consistent IRI generation patterns throughout.

**Namespace Declaration:**
- Declare **all prefixes explicitly** in both RDF and SHACL documents.
- Every namespace used (even if used only once) must be declared at the top using `@prefix`.
- Prefixes such as `iao:`, `obi:`, `matwerk:`, `qudt:`, `unit:`, etc., must be included properly.
- SHACL validation will fail if any prefix is missing or not bound.

**Validation Requirements:**

- Your SHACL file must be capable of validating the generated RDF file.
- The SHACL validation report (`sh:ValidationReport`) must indicate `sh:conforms true`.
- Validate your model using a SHACL engine (e.g., pySHACL, TopBraid SHACL Validator).
- Ensure the following alignment between RDF and SHACL:
  - Property paths match exactly.
  - Data types match (`xsd:float`, `xsd:date`, etc.).
  - Required properties are correctly marked as mandatory in SHACL.
- No missing or undefined prefixes.
- No syntax errors or validation failures are allowed.

**Important:**
- Always test the RDF and SHACL against each other before submission.
- Ensure the final model captures all quantitative and qualitative aspects of the creep test.
- Maintain interoperability with materials science domain ontologies and data standards.
"""

            # Use the selected LLM provider
            if llm_provider == "OpenAI":
                response = openai_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data_input}
                    ],
                    temperature=temperature,
                )
                content = response.choices[0].message.content
            elif llm_provider == "Anthropic (Claude)":
                response = anthropic_client.messages.create(
                    model=selected_model,
                    max_tokens=4000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": data_input}
                    ]
                )
                content = response.content[0].text
            else:  # Ollama
                # For Ollama, we'll combine system prompt and user input
                combined_prompt = f"{system_prompt}\n\nUser Input:\n{data_input}\n\nPlease provide the RDF in Turtle format and the SHACL shape in Turtle format."
                content = call_ollama_api(ollama_endpoint, selected_model, data_input, system_prompt, temperature)

        # Parse result - handle various response formats
        parts = content.split("```")
        
        # Extract RDF and SHACL code from different parts of the response
        rdf_code = ""
        shacl_code = ""
        
        # Look for turtle code blocks
        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is a code block
                if "turtle" in part.lower() or "ttl" in part.lower() or "@prefix" in part:
                    part_clean = part.replace("turtle", "").strip()
                    if not rdf_code:
                        rdf_code = part_clean
                    elif not shacl_code:
                        shacl_code = part_clean

        st.subheader("📄 RDF Output")
        st.code(rdf_code, language="turtle")
        st.download_button("⬇️ Download RDF", rdf_code, file_name="mechanical_test.ttl")

        st.subheader("📏 SHACL Shape")
        st.code(shacl_code, language="turtle")
        st.download_button("⬇️ Download SHACL", shacl_code, file_name="mechanical_test_shape.ttl")

        # ---------- SHACL Validation ----------
        st.subheader("✅ SHACL Validation")

        try:
            rdf_graph = Graph().parse(data=rdf_code, format="turtle")
            shacl_graph = Graph().parse(data=shacl_code, format="turtle")

            conforms, results_graph, results_text = validate(
                data_graph=rdf_graph,
                shacl_graph=shacl_graph,
                inference='rdfs',
                abort_on_error=False,
                meta_shacl=False,
                advanced=True,
                debug=False
            )

            if conforms:
                st.success("✅ RDF conforms to SHACL!")
            else:
                st.error("❌ RDF does NOT conform to SHACL.")
                st.text_area("SHACL Report", results_text, height=200)

        except Exception as e:
            st.error(f"Validation error: {e}")

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