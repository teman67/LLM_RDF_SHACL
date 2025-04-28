# RDF & SHACL Generator + Validator + Ontology Visualizer Using LLM

This Streamlit application is a tool for materials scientists and semantic web developers who need to convert material test data into structured RDF (Resource Description Framework) and SHACL (Shapes Constraint Language) models. The app offers several key features:
Key Features

Multi-LLM Support: Users can choose between OpenAI (GPT-4o, GPT-4o-mini, GPT-4-turbo) or Anthropic Claude (Opus, Sonnet, Haiku) models to generate their semantic models
RDF Generation: Converts raw materials science test data (particularly creep test data) into structured RDF in Turtle format
SHACL Shape Creation: Automatically generates corresponding SHACL validation shapes for the RDF data
Validation Engine: Provides immediate validation of the generated RDF against its SHACL shapes
Interactive Visualization: Creates interactive network graphs to visualize the RDF data structure
Ontology Term Suggestion: Recommends appropriate ontology terms from materials science ontologies (EMMO, MATWERK, PMDcore, NFDI, IAO, OBI, OBO, QUDT)
Export Functionality: Allows downloading of the generated RDF and SHACL files

Use Cases
This tool is particularly valuable for materials scientists working with mechanical test data who need to:

Structure their experimental data according to semantic web standards
Ensure data quality and consistency with validation rules
Map raw data to established ontologies in the materials science domain
Visualize complex data relationships for better understanding
Create interoperable datasets that align with FAIR data principles

The app serves as a bridge between raw experimental data and structured, semantic representations that can be integrated into knowledge graphs and other semantic web applications in the materials science domain.
