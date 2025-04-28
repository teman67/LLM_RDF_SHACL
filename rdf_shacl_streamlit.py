import streamlit as st
from openai import OpenAI  # updated import
import os
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# print("API Key:", api_key)  # Debugging line to check if the key is loaded

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🔬 RDF & SHACL Generator + Validator + Ontology Visualizer")

# User input
data_input = st.text_area("Paste your mechanical test data (e.g. JSON or description):", height=200)
generate = st.button("Generate RDF & SHACL")

if generate and data_input.strip():
    with st.spinner("Calling LLM to generate RDF and SHACL..."):


        system_prompt =  """
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



        # LLM Call
        response = client.chat.completions.create(
            # model="gpt-4",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_input}
            ],
            temperature=0.3,
        )

        # Parse result
        content = response.choices[0].message.content
        parts = content.split("```")
        rdf_code = parts[1].replace("turtle", "").strip()
        shacl_code = parts[3].replace("turtle", "").strip()

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


        ontology_response = client.chat.completions.create(
            # model="gpt-4",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're an ontology assistant for material science."},
                {"role": "user", "content": term_prompt}
            ],
            temperature=0.3,
        )

        st.markdown(ontology_response.choices[0].message.content)

        # ---------- Visualize with NetworkX + Pyvis ----------
        st.subheader("🌐 RDF Graph Visualization")

        def visualize_rdf(rdf_text):
            g = Graph().parse(data=rdf_text, format="turtle")
            nx_graph = nx.DiGraph()

            for s, p, o in g:
                nx_graph.add_edge(str(s), str(o), label=str(p))

            # net = Network(height="500px", width="100%", directed=True)
            net = Network(height="700px", width="100%", directed=True)

            for node in nx_graph.nodes:
                net.add_node(node, label=node)

            for u, v, d in nx_graph.edges(data=True):
                net.add_edge(u, v, label=d["label"])

            tmp_dir = tempfile.mkdtemp()
            html_path = os.path.join(tmp_dir, "graph.html")
            net.save_graph(html_path)
            return html_path

        html_file = visualize_rdf(rdf_code)
        # components.iframe(html_file, height=500)
        with open(html_file, 'r', encoding='utf-8') as f:
            graph_html = f.read()
        # components.html(graph_html, height=550, scrolling=True)
        components.html(graph_html, height=800, width=1000, scrolling=True)


