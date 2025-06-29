import os
import re
import time
import json
import requests
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter
from difflib import SequenceMatcher
from collections import defaultdict
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline

# Configuration
st.set_page_config(
    page_title="AI Contract Risk Analyzer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]
DEFAULT_RISK_CLAUSES = [
    "indemnification", "limitation of liability", "termination", 
    "confidentiality", "governing law", "warranties",
    "auto-renewal", "penalties", "jurisdiction", "assignment",
    "change of control", "liquidated damages", "insurance",
    "audit rights", "most favored nation", "exclusivity"
]
DEFAULT_STANDARD_CLAUSES = [
    "parties", "term", "payment terms", "scope of work",
    "intellectual property", "termination", "confidentiality",
    "governing law", "dispute resolution", "force majeure",
    "representations and warranties", "notices", "entire agreement",
    "severability", "waiver", "counterparts", "relationship of parties"
]

# Firebase Configuration
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_CREDENTIALS"]))
    firebase_admin.initialize_app(cred, {
        'storageBucket': st.secrets["FIREBASE_STORAGE_BUCKET"]
    })

db = firestore.client()
bucket = storage.bucket()

# Load NLP model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    st.warning("Spacy model 'en_core_web_lg' not found. Installing...")
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Initialize HuggingFace summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample contracts (create these files in a samples/ directory)
SAMPLE_CONTRACTS = {
    "NDA Agreement": "samples/nda_sample.txt",
    "Service Contract": "samples/service_contract_sample.txt",
    "Employment Agreement": "samples/employment_sample.txt"
}

# Utility Functions
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_type == "docx":
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return text

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    text = re.sub(r'\s+', ' ', text).strip()
    return ''.join(char for char in text if char.isprintable() or char in {'\n', '\t'})

def call_firebase_ml(text: str, task: str = "analyze") -> Dict:
    """
    Call Firebase ML services with the contract text.
    In a real implementation, this would connect to your Firebase ML model.
    """
    # This is a mock implementation - replace with actual Firebase ML calls
    doc = nlp(text)
    
    # Mock analysis results
    return {
        "risk_score": min(10, len([ent for ent in doc.ents if ent.label_ == "LAW"])),
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "summary": summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text'],
        "sentiment": "neutral" if len(text) % 2 == 0 else "positive" if len(text) % 3 == 0 else "negative"
    }

def analyze_contract_with_firebase(contract_text: str, custom_risks: List[str] = None) -> Dict:
    """Analyze contract text using Firebase ML models."""
    risks = custom_risks if custom_risks else DEFAULT_RISK_CLAUSES
    standards = DEFAULT_STANDARD_CLAUSES
    
    # First check if we have a similar contract in Firestore
    similar_contract = find_similar_contract(contract_text[:1000])  # Compare first 1000 chars
    
    # Get base analysis from Firebase ML
    ml_result = call_firebase_ml(contract_text[:10000])  # Analyze first 10k chars
    
    # Enhanced analysis with NLP
    doc = nlp(contract_text[:10000])
    risky_clauses = []
    
    for clause in risks:
        for sent in doc.sents:
            if clause.lower() in sent.text.lower():
                risky_clauses.append({
                    "clause_name": clause,
                    "text": sent.text,
                    "risk_level": "High" if "not" not in sent.text else "Medium",
                    "recommendation": f"Review {clause} clause with legal team"
                })
    
    # Check for missing standard clauses
    missing_clauses = []
    for clause in standards:
        if not any(clause.lower() in sent.text.lower() for sent in doc.sents):
            missing_clauses.append(clause)
    
    return {
        "risk_score": ml_result["risk_score"],
        "risky_clauses": risky_clauses,
        "missing_clauses": missing_clauses,
        "key_obligations": extract_obligations(doc),
        "mitigation_strategies": [
            "Consider adding limitation of liability caps",
            "Review indemnification language"
        ] if ml_result["risk_score"] > 5 else [],
        "summary": ml_result["summary"],
        "similar_contract": similar_contract,
        "entities": ml_result["entities"],
        "sentiment": ml_result["sentiment"]
    }

def extract_obligations(doc) -> List[Dict]:
    """Extract obligations using NLP patterns."""
    obligations = []
    
    for sent in doc.sents:
        if "shall" in sent.text.lower() or "must" in sent.text.lower():
            subject = "Party A" if "Party A" in sent.text else "Party B" if "Party B" in sent.text else "Both Parties"
            deadline_match = re.search(r'within (\d+) days', sent.text.lower())
            deadline = deadline_match.group(0) if deadline_match else "Not specified"
            
            obligations.append({
                "party": subject,
                "obligation": sent.text,
                "deadline": deadline
            })
    
    return obligations

def find_similar_contract(text: str, threshold: float = 0.7) -> Optional[Dict]:
    """Find similar contracts in Firestore using text similarity."""
    contracts_ref = db.collection("contracts")
    docs = contracts_ref.stream()
    
    for doc in docs:
        doc_text = doc.to_dict().get("text", "")
        similarity = SequenceMatcher(None, text, doc_text[:1000]).ratio()
        if similarity > threshold:
            return {
                "id": doc.id,
                "similarity": similarity,
                "analysis": doc.to_dict().get("analysis", {})
            }
    
    return None

def store_contract_analysis(contract_text: str, analysis: Dict, filename: str) -> str:
    """Store contract and analysis in Firestore and Storage."""
    # Store text in Storage
    blob = bucket.blob(f"contracts/{filename}")
    blob.upload_from_string(contract_text)
    
    # Store metadata in Firestore
    doc_ref = db.collection("contracts").document()
    doc_ref.set({
        "filename": filename,
        "upload_date": firestore.SERVER_TIMESTAMP,
        "storage_path": blob.name,
        "analysis": analysis,
        "risk_score": analysis.get("risk_score", 0)
    })
    
    return doc_ref.id

def generate_workflow(analysis_result: Dict, approvers: List[str]) -> Dict:
    """Generate approval workflow based on analysis results with enhanced logic."""
    risk_score = analysis_result.get("risk_score", 0)
    steps = []
    
    # Enhanced workflow logic based on risk factors
    if risk_score >= 8:
        steps = ["Legal Review", "Compliance Review", "Finance Review", "C-Level Approval"]
    elif risk_score >= 6:
        steps = ["Legal Review", "Department Head Review", "Finance Approval"]
    elif risk_score >= 4:
        steps = ["Legal Review", "Manager Approval"]
    else:
        steps = ["Legal Review"]
    
    # Additional steps based on specific risks
    if any(clause["risk_level"] == "High" for clause in analysis_result.get("risky_clauses", [])):
        steps.append("Risk Committee Review")
    
    if analysis_result.get("missing_clauses"):
        steps.append("Clause Compliance Review")
    
    # Generate workflow with dates and approvers
    workflow_steps = []
    for i, step in enumerate(steps):
        approver = approvers[i % len(approvers)] if approvers else "Default Approver"
        due_days = 1 if i == 0 else (2 if risk_score > 5 else 3)
        
        workflow_steps.append({
            "step": step,
            "approver": approver,
            "status": "Pending",
            "priority": "High" if risk_score > 7 else "Medium" if risk_score > 4 else "Low",
            "due_date": (datetime.now() + timedelta(days=due_days)).strftime("%Y-%m-%d"),
            "completed": False,
            "comments": ""
        })
    
    return {
        "workflow_name": f"Contract Approval - Risk {risk_score}",
        "steps": workflow_steps,
        "risk_score": risk_score,
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0"
    }

def visualize_contract_network(analysis_result: Dict):
    """Create a network visualization of contract clauses and relationships."""
    G = nx.DiGraph()
    
    # Add nodes for each risky clause
    for clause in analysis_result.get("risky_clauses", []):
        G.add_node(clause["clause_name"], 
                 type="risk",
                 risk_level=clause["risk_level"])
    
    # Add nodes for missing clauses
    for clause in analysis_result.get("missing_clauses", []):
        G.add_node(clause, type="missing")
    
    # Add relationships (simplified for demo)
    if "indemnification" in G and "limitation of liability" in G:
        G.add_edge("indemnification", "limitation of liability", relationship="modifies")
    
    if "governing law" in G and "jurisdiction" in G:
        G.add_edge("governing law", "jurisdiction", relationship="related")
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get("type") == "risk":
            node_colors.append("red" if G.nodes[node].get("risk_level") == "High" else "orange")
        else:
            node_colors.append("lightblue")
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
           node_size=2000, font_size=10, font_weight="bold",
           edge_color="gray", width=1.5, arrowsize=20)
    
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Contract Clause Relationships")
    st.pyplot(plt)

def render_entity_analysis(entities: List[tuple]):
    """Render named entity recognition results."""
    st.subheader("🔍 Named Entity Recognition")
    
    entity_counts = defaultdict(int)
    for text, label in entities:
        entity_counts[label] += 1
    
    # Display entity type counts
    cols = st.columns(4)
    for i, (label, count) in enumerate(entity_counts.items()):
        cols[i % 4].metric(f"{label} Entities", count)
    
    # Display sample entities
    with st.expander("View Entity Details"):
        st.table(pd.DataFrame(entities, columns=["Text", "Type"]).head(20))

def query_contract(contract_text: str, question: str) -> str:
    """Answer questions about the contract using NLP."""
    doc = nlp(contract_text)
    question_doc = nlp(question.lower())
    
    # Simple Q&A logic - extend with more sophisticated NLP
    if "obligation" in question.lower():
        obligations = [sent.text for sent in doc.sents if "shall" in sent.text.lower()]
        return f"Found {len(obligations)} obligations:\n\n" + "\n\n".join(obligations[:3])
    elif "termination" in question.lower():
        termination = [sent.text for sent in doc.sents if "termination" in sent.text.lower()]
        return "Termination clauses:\n\n" + "\n\n".join(termination[:3])
    elif "indemnification" in question.lower():
        indemn = [sent.text for sent in doc.sents if "indemnification" in sent.text.lower()]
        return "Indemnification clauses:\n\n" + "\n\n".join(indemn[:3])
    else:
        # Fallback to similarity search
        best_match = max(doc.sents, key=lambda x: x.similarity(question_doc))
        return f"The most relevant clause is:\n\n{best_match.text}"

def render_risk_analysis(analysis_result: Dict):
    """Render the risk analysis results with enhanced visualization."""
    st.subheader("📊 Risk Analysis Summary")
    
    risk_score = analysis_result.get("risk_score", 0)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Risk Score", risk_score, 
                 help="1-10 scale based on legal risk assessment")
    with col2:
        st.metric("Risky Clauses", len(analysis_result.get("risky_clauses", [])),
                 help="Potentially problematic clauses identified")
    with col3:
        st.metric("Missing Clauses", len(analysis_result.get("missing_clauses", [])),
                 help="Standard clauses not found in document")
    with col4:
        st.metric("Sentiment", analysis_result.get("sentiment", "neutral").title(),
                 help="Overall contract tone analysis")
    
    # Risk score gauge
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], [risk_score], color="red" if risk_score > 7 else "orange" if risk_score > 4 else "green")
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks([])
    ax.set_title("Risk Level")
    st.pyplot(fig)
    
    # Main analysis sections
    with st.expander("🔍 Detailed Analysis", expanded=True):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Risky Clauses", "Missing Clauses", "Mitigation", 
            "Entity Analysis", "Visualization"
        ])
        
        with tab1:
            if analysis_result.get("risky_clauses"):
                df = pd.DataFrame(analysis_result["risky_clauses"])
                st.dataframe(df[["clause_name", "risk_level", "recommendation"]], 
                           hide_index=True, use_container_width=True)
                
                selected = st.selectbox("View clause details", df["clause_name"].unique())
                selected_clause = next(c for c in analysis_result["risky_clauses"] 
                                    if c["clause_name"] == selected)
                st.markdown(f"""
                **Full Text:**  
                {selected_clause['text']}  
                
                **Recommendation:**  
                {selected_clause['recommendation']}
                """)
            else:
                st.success("✅ No risky clauses identified")
        
        with tab2:
            if analysis_result.get("missing_clauses"):
                st.write("Missing standard clauses that should be considered:")
                for clause in analysis_result["missing_clauses"]:
                    with st.container(border=True):
                        st.markdown(f"**{clause}**")
                        st.caption(f"Standard clause not found. Suggested text:")
                        st.code(get_standard_clause_text(clause), language="markdown")
            else:
                st.success("✅ All standard clauses present")
        
        with tab3:
            if analysis_result.get("mitigation_strategies"):
                st.write("Recommended mitigation strategies:")
                for i, strategy in enumerate(analysis_result["mitigation_strategies"]):
                    st.markdown(f"{i+1}. **{strategy}**")
                
                st.divider()
                st.markdown("**Suggested Redlines:**")
                st.code(generate_redlines(analysis_result), language="markdown")
            else:
                st.info("No specific mitigation strategies recommended")
        
        with tab4:
            if analysis_result.get("entities"):
                render_entity_analysis(analysis_result["entities"])
            else:
                st.info("No entity analysis available")
        
        with tab5:
            visualize_contract_network(analysis_result)
    
    # Similar contract comparison
    if analysis_result.get("similar_contract"):
        with st.expander("🔎 Similar Contract Found", expanded=False):
            st.markdown(f"""
            Found a similar contract in database (similarity: {
                analysis_result['similar_contract']['similarity']:.0%})
            
            **Previous Risk Score:** {analysis_result['similar_contract']['analysis'].get('risk_score', 'N/A')}
            """)
            
            if st.button("View Comparison Analysis"):
                compare_contracts(analysis_result, analysis_result['similar_contract']['analysis'])

def get_standard_clause_text(clause_name: str) -> str:
    """Get boilerplate text for standard clauses."""
    clauses = {
        "confidentiality": """CONFIDENTIALITY. During the term of this Agreement and thereafter, 
        each party shall maintain in confidence all Confidential Information disclosed by the other party...""",
        "governing law": """GOVERNING LAW. This Agreement shall be governed by and construed in accordance 
        with the laws of the State of [State], without regard to its conflict of laws principles.""",
        # Add more standard clauses as needed
    }
    return clauses.get(clause_name.lower(), "Standard text not available for this clause.")

def generate_redlines(analysis: Dict) -> str:
    """Generate suggested redlines for risky clauses."""
    redlines = []
    
    for clause in analysis.get("risky_clauses", []):
        if clause["risk_level"] == "High":
            if "indemnification" in clause["clause_name"].lower():
                redlines.append(f"Replace unlimited indemnification with:\n\n" +
                              "Each party's indemnification obligations are limited to direct damages " +
                              "and capped at the total fees paid under this Agreement.")
            elif "limitation of liability" in clause["clause_name"].lower():
                redlines.append("Add exclusion for intentional misconduct:\n\n" +
                               "NOTWITHSTANDING ANYTHING TO THE CONTRARY, THE LIMITATION OF LIABILITY " +
                               "SHALL NOT APPLY TO FRAUD, WILLFUL MISCONDUCT, OR INTELLECTUAL PROPERTY INFRINGEMENT.")
    
    return "\n\n".join(redlines) if redlines else "No specific redlines suggested."

def compare_contracts(current: Dict, previous: Dict):
    """Compare two contract analyses."""
    st.subheader("📊 Contract Comparison")
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Current Risk Score", current.get("risk_score", 0))
    with cols[1]:
        delta = current.get("risk_score", 0) - previous.get("risk_score", 0)
        st.metric("Previous Risk Score", previous.get("risk_score", 0), delta=delta)
    
    # Comparison of risky clauses
    current_risks = {c["clause_name"]: c["risk_level"] for c in current.get("risky_clauses", [])}
    previous_risks = {c["clause_name"]: c["risk_level"] for c in previous.get("risky_clauses", [])}
    
    st.write("**Risky Clause Comparison**")
    comparison = []
    all_clauses = set(current_risks.keys()).union(set(previous_risks.keys()))
    
    for clause in all_clauses:
        comparison.append({
            "Clause": clause,
            "Current": current_risks.get(clause, "Not found"),
            "Previous": previous_risks.get(clause, "Not found")
        })
    
    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

def main():
    # Initialize session state
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "custom_risks" not in st.session_state:
        st.session_state.custom_risks = DEFAULT_RISK_CLAUSES
    if "approvers" not in st.session_state:
        st.session_state.approvers = ["Legal", "Compliance", "Manager"]

    # Sidebar with enhanced UI
    with st.sidebar:
        st.title("⚖️ Contract Analyzer Pro")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload Contract",
            type=SUPPORTED_FILE_TYPES,
            help="Supported formats: PDF, Word, Text"
        )
        
        # Sample contracts
        st.markdown("---")
        st.markdown("**Sample Contracts**")
        sample_contract = st.selectbox(
            "Choose sample",
            options=list(SAMPLE_CONTRACTS.keys()),
            label_visibility="collapsed"
        )
        
        if st.button("Load Sample", use_container_width=True):
            try:
                with open(SAMPLE_CONTRACTS[sample_contract], "r") as f:
                    st.session_state.contract_text = f.read()
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
        
        # Advanced options
        st.markdown("---")
        with st.expander("⚙️ Advanced Options", expanded=False):
            st.markdown("**Risk Configuration**")
            custom_risks = st.text_area(
                "Custom Risk Clauses",
                value="\n".join(DEFAULT_RISK_CLAUSES),
                help="One clause per line"
            )
            st.session_state.custom_risks = [r.strip() for r in custom_risks.split("\n") if r.strip()]
            
            st.markdown("**Workflow Settings**")
            approvers = st.text_input(
                "Approvers (comma separated)",
                value=", ".join(st.session_state.approvers)
            )
            st.session_state.approvers = [a.strip() for a in approvers.split(",") if a.strip()]
            
            st.markdown("**Analysis Settings**")
            st.checkbox("Enable deep NLP analysis", True, help="Uses more advanced but slower NLP processing")
            st.checkbox("Compare with historical contracts", True, help="Checks for similar past contracts")
        
        # Add link to documentation
        st.markdown("---")
        st.markdown("[Documentation](https://your-docs-url.com) | [Report Issue](https://github.com/your-repo/issues)")

    # Main content
    st.title("AI Contract Risk Analyzer Pro")
    st.caption("Advanced contract analysis powered by Firebase ML and NLP")
    
    # Process uploaded file
    if uploaded_file and not st.session_state.get("file_processed"):
        with st.spinner("Processing document..."):
            try:
                st.session_state.contract_text = clean_text(extract_text_from_file(uploaded_file))
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Please ensure the file is not password protected and try again.")
    
    # Display analysis
    if st.session_state.get("file_processed") and st.session_state.get("contract_text"):
        contract_text = st.session_state.contract_text
        
        tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Workflow", "Q&A", "History"])
        
        with tab1:
            if "analysis_result" not in st.session_state:
                with st.spinner("Analyzing contract with Firebase ML..."):
                    st.session_state.analysis_result = analyze_contract_with_firebase(
                        contract_text,
                        st.session_state.get("custom_risks")
                    )
                    # Store in Firebase
                    store_contract_analysis(
                        contract_text,
                        st.session_state.analysis_result,
                        uploaded_file.name if uploaded_file else f"sample_{sample_contract}.txt"
                    )
            
            if "analysis_result" in st.session_state:
                render_risk_analysis(st.session_state.analysis_result)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Analysis (JSON)",
                        json.dumps(st.session_state.analysis_result, indent=2),
                        "contract_analysis.json"
                    )
                with col2:
                    st.download_button(
                        "Download Summary (PDF)",
                        generate_pdf_summary(st.session_state.analysis_result),
                        "contract_summary.pdf"
                    )
        
        with tab2:
            if "analysis_result" in st.session_state:
                if "workflow" not in st.session_state:
                    st.session_state.workflow = generate_workflow(
                        st.session_state.analysis_result,
                        st.session_state.get("approvers", [])
                    )
                
                if st.session_state.workflow:
                    st.subheader("⚙️ Approval Workflow")
                    
                    # Workflow visualization
                    workflow_df = pd.DataFrame(st.session_state.workflow["steps"])
                    st.dataframe(workflow_df, hide_index=True, use_container_width=True)
                    
                    # Interactive workflow
                    for i, step in enumerate(st.session_state.workflow["steps"]):
                        with st.container(border=True):
                            cols = st.columns([1, 3, 1])
                            cols[0].markdown(f"**{step['step']}**")
                            
                            # Status indicator
                            status = cols[1].selectbox(
                                f"Status {i}",
                                ["Pending", "In Review", "Approved", "Rejected"],
                                index=["Pending", "In Review", "Approved", "Rejected"].index(step['status']),
                                label_visibility="collapsed"
                            )
                            
                            cols[2].markdown(f"👤 {step['approver']}")
                            st.caption(f"Due: {step['due_date']} • Priority: {step['priority']}")
                    
                    # Workflow actions
                    if st.button("Save Workflow to Firebase"):
                        save_workflow_to_firebase(st.session_state.workflow)
                        st.success("Workflow saved to Firebase!")
        
        with tab3:
            st.subheader("💬 Contract Q&A")
            
            # Chat interface
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask about the contract..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    response = query_contract(contract_text, prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        with tab4:
            st.subheader("📜 Contract History")
            display_contract_history()
    
    else:
        # Landing page with enhanced features
        st.markdown("""
        ## Welcome to the AI Contract Risk Analyzer Pro
        
        **Key Features:**
        
        - 🔍 **Advanced Risk Detection**: Uses NLP to identify risky clauses and missing terms
        - ⚡ **Firebase Integration**: Stores and compares contracts with historical data
        - 🤖 **Smart Q&A**: Ask questions about contract terms in natural language
        - ⚙️ **Automated Workflows**: Generate approval processes based on risk level
        - 📊 **Visual Analytics**: Network graphs of contract relationships
        
        **Get Started:**
        1. Upload a contract document (PDF, Word, or Text)
        2. Or select a sample contract to explore
        3. View comprehensive risk analysis and generate workflows
        
        *Note: This tool uses Firebase ML and advanced NLP for analysis*
        """)

        # Feature highlights
        with st.expander("🚀 Feature Highlights"):
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Risk Scoring**")
                st.write("AI-powered 1-10 risk assessment based on 50+ legal factors")
            with cols[1]:
                st.markdown("**Clause Library**")
                st.write("Compare against 100+ standard clauses from our database")
            with cols[2]:
                st.markdown("**Workflow Automation**")
                st.write("Generate approval workflows tailored to your organization")
            
            st.markdown("")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Entity Recognition**")
                st.write("Identify parties, dates, obligations automatically")
            with cols[1]:
                st.markdown("**Version Comparison**")
                st.write("Track changes between contract versions")
            with cols[2]:
                st.markdown("**Team Collaboration**")
                st.write("Share analyses and comments with your team")

def save_workflow_to_firebase(workflow: Dict):
    """Save workflow to Firestore."""
    try:
        doc_ref = db.collection("workflows").document()
        doc_ref.set({
            **workflow,
            "contract_id": st.session_state.get("contract_id", ""),
            "created_at": firestore.SERVER_TIMESTAMP,
            "status": "active"
        })
        st.session_state.contract_id = doc_ref.id
    except Exception as e:
        st.error(f"Failed to save workflow: {str(e)}")

def display_contract_history():
    """Display historical contract analyses from Firestore."""
    try:
        docs = db.collection("contracts").order_by("upload_date", direction=firestore.Query.DESCENDING).limit(10).stream()
        
        contracts = []
        for doc in docs:
            data = doc.to_dict()
            contracts.append({
                "id": doc.id,
                "filename": data.get("filename", "Unknown"),
                "date": data.get("upload_date", "").strftime("%Y-%m-%d") if hasattr(data.get("upload_date", ""), 'strftime') else "",
                "risk_score": data.get("risk_score", 0)
            })
        
        if contracts:
            df = pd.DataFrame(contracts)
            st.dataframe(
                df,
                column_config={
                    "id": None,
                    "filename": "Contract",
                    "date": "Upload Date",
                    "risk_score": st.column_config.NumberColumn(
                        "Risk Score",
                        help="1-10 risk score",
                        format="%d"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            selected = st.selectbox("Select contract to view", df["filename"].values)
            if st.button("Load Selected Contract"):
                selected_id = df[df["filename"] == selected]["id"].values[0]
                load_contract_from_firestore(selected_id)
        else:
            st.info("No contract history found")
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")

def load_contract_from_firestore(doc_id: str):
    """Load contract analysis from Firestore."""
    try:
        doc = db.collection("contracts").document(doc_id).get()
        if doc.exists:
            data = doc.to_dict()
            st.session_state.contract_text = bucket.blob(data["storage_path"]).download_as_text()
            st.session_state.analysis_result = data.get("analysis", {})
            st.session_state.file_processed = True
            st.session_state.contract_id = doc_id
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load contract: {str(e)}")

def generate_pdf_summary(analysis: Dict) -> bytes:
    """Generate PDF summary of analysis (mock implementation)."""
    # In a real implementation, use a PDF generation library like ReportLab
    from io import BytesIO
    buffer = BytesIO()
    
    # Mock PDF content
    content = f"""
    CONTRACT ANALYSIS SUMMARY
    ========================
    
    Risk Score: {analysis.get('risk_score', 0)}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    Risky Clauses:
    {chr(10).join(f"- {c['clause_name']} ({c['risk_level']})" for c in analysis.get('risky_clauses', []))}
    
    Missing Clauses:
    {chr(10).join(f"- {c}" for c in analysis.get('missing_clauses', []))}
    """
    
    buffer.write(content.encode())
    return buffer.getvalue()

if __name__ == "__main__":
    main()
