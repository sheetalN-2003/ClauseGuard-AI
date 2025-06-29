import os
import re
import time
import json
import base64
import requests
import pandas as pd
import streamlit as st
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from PyPDF2 import PdfReader
from docx import Document
from datetime import datetime, timedelta

# Configuration
st.set_page_config(
    page_title="AI Contract Risk Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]
DEFAULT_RISK_CLAUSES = [
    "indemnification", "limitation of liability", "termination", 
    "confidentiality", "governing law", "warranties",
    "auto-renewal", "penalties", "jurisdiction"
]
DEFAULT_STANDARD_CLAUSES = [
    "parties", "term", "payment terms", "scope of work",
    "intellectual property", "termination", "confidentiality",
    "governing law", "dispute resolution", "force majeure"
]

# IBM Cloud Configuration
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
GRANITE_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-01"
DEFAULT_MODEL = "ibm/granite-13b-instruct-v1"

# Sample data for demonstration
SAMPLE_CONTRACTS = {
    "NDA Agreement": "samples/nda_sample.txt",
    "Service Contract": "samples/service_contract_sample.txt",
    "Employment Agreement": "samples/employment_sample.txt"
}

# Authentication Functions
def get_iam_token() -> str:
    """Fetch IAM token using API key from Streamlit secrets"""
    if "iam_token" in st.session_state and st.session_state.iam_token_expiry > time.time():
        return st.session_state.iam_token
    
    try:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": st.secrets["IAM_API_KEY"]
        }
        
        response = requests.post(
            IAM_TOKEN_URL,
            headers=headers,
            data=data,
            timeout=10
        )
        response.raise_for_status()
        
        st.session_state.iam_token = response.json().get("access_token", "")
        st.session_state.iam_token_expiry = time.time() + 3500  # 58 minutes for safety
        
        if not st.session_state.iam_token:
            st.error("Failed to obtain IAM token: Empty response")
            return ""
            
        return st.session_state.iam_token
        
    except Exception as e:
        st.error(f"IAM token fetch failed: {str(e)}")
        return ""

# Utility Functions
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages])
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
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in {'\n', '\t'})
    return text

def call_granite_api(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call IBM Granite API with IAM authentication."""
    iam_token = get_iam_token()
    if not iam_token:
        return ""
    
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model_id": model,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 2000,
            "min_new_tokens": 10,
            "temperature": 0.7,
            "repetition_penalty": 1.1
        },
        "project_id": st.secrets["GRANITE_INSTANCE_ID"]
    }
    
    try:
        with st.spinner("Analyzing with Granite model..."):
            response = requests.post(
                GRANITE_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json().get("results", [{}])
            if not result:
                st.error("Empty response from Granite API")
                return ""
                
            return result[0].get("generated_text", "")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Granite API request failed: {str(e)}")
        return ""
    except json.JSONDecodeError:
        st.error("Failed to decode Granite API response")
        return ""
    except Exception as e:
        st.error(f"Unexpected error calling Granite API: {str(e)}")
        return ""

def analyze_contract_with_granite(contract_text: str, custom_risks: List[str] = None) -> Dict:
    """Analyze contract text using Granite models."""
    risks = custom_risks if custom_risks else DEFAULT_RISK_CLAUSES
    standards = DEFAULT_STANDARD_CLAUSES
    
    # Prepare the analysis prompt
    prompt = f"""
    Analyze the following contract text and provide a detailed risk assessment.
    
    Contract Text:
    {contract_text[:10000]}  # Limiting to first 10k chars for demo
    
    Instructions:
    1. Identify and highlight any risky clauses related to: {', '.join(risks)}
    2. Check for missing standard clauses: {', '.join(standards)}
    3. Extract key obligations, deadlines, and important dates
    4. Provide a risk score from 1-10 (10 being highest risk)
    5. Suggest mitigation strategies for high-risk clauses
    
    Format your response as a JSON object with these keys:
    - "risk_score": number
    - "risky_clauses": list of objects with "clause_name", "text", "risk_level", "recommendation"
    - "missing_clauses": list
    - "key_obligations": list of objects with "party", "obligation", "deadline"
    - "summary": string
    - "mitigation_strategies": list of strings
    
    Return ONLY the JSON object, no additional text or explanation.
    """
    
    response = call_granite_api(prompt)
    
    try:
        # Try to extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        return json.loads(json_str)
    except json.JSONDecodeError:
        st.error("Failed to parse Granite API response as JSON. Showing raw response.")
        return {"raw_response": response}

def generate_workflow(analysis_result: Dict, approvers: List[str]) -> Dict:
    """Generate approval workflow based on analysis results."""
    risk_score = analysis_result.get("risk_score", 0)
    risky_clauses = analysis_result.get("risky_clauses", [])
    missing_clauses = analysis_result.get("missing_clauses", [])
    
    # Determine workflow steps based on risk
    if risk_score >= 7:
        steps = ["Legal Review", "Compliance Review", "Senior Management Approval"]
    elif risk_score >= 4:
        steps = ["Legal Review", "Department Head Approval"]
    else:
        steps = ["Legal Review"]
    
    # Add steps for missing clauses
    if missing_clauses:
        steps.append("Clause Addition Review")
    
    # Assign approvers
    workflow_steps = []
    for i, step in enumerate(steps):
        approver = approvers[i % len(approvers)] if approvers else "Default Approver"
        workflow_steps.append({
            "step": step,
            "approver": approver,
            "status": "Pending",
            "required": "Mandatory" if i == 0 or risk_score >= 7 else "Recommended",
            "due_date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
        })
    
    return {
        "workflow_name": f"Contract Approval - Risk {risk_score}",
        "steps": workflow_steps,
        "risk_score": risk_score,
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_contract_summary(contract_text: str) -> str:
    """Generate a concise summary of the contract."""
    prompt = f"""
    Summarize the following contract text in 3-5 bullet points focusing on:
    - Key parties involved
    - Main obligations
    - Important dates and durations
    - Termination conditions
    - Any unusual or notable clauses
    
    Contract Text:
    {contract_text[:8000]}  # Limiting length for demo
    
    Provide the summary in clear, concise bullet points suitable for an executive.
    """
    
    return call_granite_api(prompt)

def query_contract(contract_text: str, query: str) -> str:
    """Answer questions about the contract content."""
    prompt = f"""
    You are a contract analysis assistant. Answer the following question about the provided contract text.
    
    Question: {query}
    
    Contract Text:
    {contract_text[:8000]}  # Limiting length for demo
    
    Provide a concise answer with relevant clause references if possible.
    If the information is not in the contract, state that clearly.
    """
    
    return call_granite_api(prompt)

# UI Components
def render_risk_analysis(analysis_result: Dict):
    """Render the risk analysis results."""
    st.subheader("📊 Risk Analysis Summary")
    
    risk_score = analysis_result.get("risk_score", 0)
    risk_color = "red" if risk_score >= 7 else "orange" if risk_score >= 4 else "green"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Risk Score", risk_score, help="1-10 scale (10 = highest risk)")
    
    with col2:
        risky_count = len(analysis_result.get("risky_clauses", []))
        st.metric("Risky Clauses Identified", risky_count)
    
    with col3:
        missing_count = len(analysis_result.get("missing_clauses", []))
        st.metric("Missing Standard Clauses", missing_count)
    
    # Risk visualization
    with st.expander("🔍 Detailed Risk Breakdown", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["Risky Clauses", "Missing Clauses", "Key Obligations", "Mitigation Strategies"])
        
        with tab1:
            if analysis_result.get("risky_clauses"):
                for clause in analysis_result["risky_clauses"]:
                    st.markdown(f"""
                    **{clause.get('clause_name', 'Unnamed Clause')}**  
                    *Risk Level: {clause.get('risk_level', 'Unknown')}*  
                    {clause.get('text', '')}  
                    🛡️ **Recommendation:** {clause.get('recommendation', 'None provided')}
                    """)
                    st.divider()
            else:
                st.info("No risky clauses identified")
        
        with tab2:
            if analysis_result.get("missing_clauses"):
                st.write("The following standard clauses are missing from the contract:")
                for clause in analysis_result["missing_clauses"]:
                    st.markdown(f"- {clause}")
            else:
                st.success("All standard clauses are present in the contract")
        
        with tab3:
            if analysis_result.get("key_obligations"):
                df = pd.DataFrame(analysis_result["key_obligations"])
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.info("No specific obligations identified")
        
        with tab4:
            if analysis_result.get("mitigation_strategies"):
                for strategy in analysis_result["mitigation_strategies"]:
                    st.markdown(f"- {strategy}")
            else:
                st.info("No specific mitigation strategies recommended")

def render_workflow(workflow: Dict):
    """Render the generated workflow."""
    st.subheader("⚙️ Approval Workflow")
    
    if not workflow.get("steps"):
        st.warning("No workflow steps generated")
        return
    
    # Gantt-like visualization
    st.write("### Workflow Timeline")
    for step in workflow["steps"]:
        due_date = datetime.strptime(step["due_date"], "%Y-%m-%d")
        days_until = (due_date - datetime.now()).days
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"**{step['step']}**")
        with col2:
            progress = min(100, max(0, 100 - (days_until * 10)))
            st.progress(progress, text=f"Due in {days_until} days")
        with col3:
            st.markdown(f"👤 {step['approver']}")
        
        st.caption(f"Status: {step['status']} • {step['required']}")
        st.divider()
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Workflow", use_container_width=True):
            st.success("Workflow approved and notifications sent to approvers")
    with col2:
        if st.button("✏️ Modify Workflow", use_container_width=True):
            st.session_state.editing_workflow = True

def render_contract_chat(contract_text: str):
    """Render the interactive contract Q&A interface."""
    st.subheader("💬 Contract Q&A")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the contract..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing contract..."):
                response = query_contract(contract_text, prompt)
                st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Main App
def main():
    # Initialize session state variables
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "iam_token" not in st.session_state:
        st.session_state.iam_token = ""
        st.session_state.iam_token_expiry = 0

    # Sidebar
    with st.sidebar:
        st.title("⚖️ Contract Analyzer")
        st.markdown("""
        Upload contracts to:
        - 🔍 Identify risks
        - 📝 Summarize key terms
        - ⚡ Generate approval workflows
        """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Contract",
            type=SUPPORTED_FILE_TYPES,
            help="Supported formats: PDF, Word, Text"
        )
        
        # Sample contracts
        st.markdown("---")
        st.markdown("**Or try a sample contract:**")
        sample_contract = st.selectbox(
            "Select sample contract",
            options=list(SAMPLE_CONTRACTS.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        if st.button("Load Sample Contract", use_container_width=True):
            try:
                with open(SAMPLE_CONTRACTS[sample_contract], "r") as f:
                    st.session_state.contract_text = f.read()
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample contract: {str(e)}")
        
        # Advanced options
        st.markdown("---")
        with st.expander("⚙️ Advanced Options"):
            custom_risks = st.text_area(
                "Custom Risk Clauses to Detect",
                value="\n".join(DEFAULT_RISK_CLAUSES),
                help="Enter one clause per line"
            )
            st.session_state.custom_risks = [r.strip() for r in custom_risks.split("\n") if r.strip()]
            
            approvers = st.text_input(
                "Approvers (comma separated)",
                value="Legal Team, Compliance Officer, Department Head",
                help="People to include in the workflow"
            )
            st.session_state.approvers = [a.strip() for a in approvers.split(",") if a.strip()]
    
    # Main content
    st.title("AI Contract Risk Analyzer & Workflow Automator")
    
    # Check for required secrets
    if not all(key in st.secrets for key in ["IAM_API_KEY", "GRANITE_INSTANCE_ID"]):
        st.error("""
        Missing required configuration. Please ensure you have:
        - IAM_API_KEY
        - GRANITE_INSTANCE_ID
        in your Streamlit secrets.
        """)
        return
    
    # Process uploaded file
    if uploaded_file and not st.session_state.get("file_processed"):
        with st.spinner("Processing contract..."):
            try:
                st.session_state.contract_text = clean_text(extract_text_from_file(uploaded_file))
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Display analysis if file is processed
    if st.session_state.get("file_processed") and st.session_state.get("contract_text"):
        contract_text = st.session_state.contract_text
        
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "⚠️ Risk Analysis", "⚙️ Workflow", "💬 Q&A"])
        
        with tab1:
            st.subheader("Contract Summary")
            with st.spinner("Generating summary..."):
                summary = generate_contract_summary(contract_text)
                st.markdown(summary)
            
            st.download_button(
                label="Download Summary",
                data=summary.encode("utf-8"),
                file_name="contract_summary.txt",
                mime="text/plain"
            )
            
            st.markdown("---")
            with st.expander("📜 View Full Contract Text"):
                st.text(contract_text[:5000] + "..." if len(contract_text) > 5000 else contract_text)
        
        with tab2:
            if "analysis_result" not in st.session_state:
                with st.spinner("Analyzing contract for risks..."):
                    st.session_state.analysis_result = analyze_contract_with_granite(
                        contract_text,
                        st.session_state.get("custom_risks")
                    )
                    if not st.session_state.analysis_result:
                        st.error("Failed to analyze contract. Please check your API configuration.")
            
            if "analysis_result" in st.session_state:
                render_risk_analysis(st.session_state.analysis_result)
        
        with tab3:
            if "workflow" not in st.session_state and "analysis_result" in st.session_state:
                with st.spinner("Generating approval workflow..."):
                    st.session_state.workflow = generate_workflow(
                        st.session_state.analysis_result,
                        st.session_state.get("approvers", [])
                    )
            
            if "workflow" in st.session_state:
                render_workflow(st.session_state.workflow)
                
                # Export workflow
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Workflow (JSON)",
                        data=json.dumps(st.session_state.workflow, indent=2),
                        file_name="contract_workflow.json",
                        mime="application/json"
                    )
                with col2:
                    if st.button("Send to Approval System", use_container_width=True):
                        st.success("Workflow sent to approval system")
        
        with tab4:
            render_contract_chat(contract_text)
    
    elif not st.session_state.get("file_processed"):
        # Show welcome/instructions if no file uploaded
        st.markdown("""
        ## Welcome to the AI Contract Risk Analyzer
        
        This tool helps legal teams and business professionals:
        
        - 🔍 **Automatically identify risks** in contracts
        - 📝 **Summarize key obligations** and deadlines
        - ⚡ **Generate approval workflows** based on risk level
        - 💬 **Ask questions** about contract terms
        
        ### Getting Started
        
        1. Upload a contract document (PDF, Word, or Text)
        2. Or select a sample contract from the sidebar
        3. View the analysis results across the tabs
        
        *Note: This is a demo using IBM Granite models. No actual contracts are stored.*
        """)

if __name__ == "__main__":
    main()
