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
from typing import List, Dict

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
GRANITE_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-05-01"
DEFAULT_MODEL = "ibm/granite-13b-instruct-v2"  # Updated recommended model

# Sample contracts (create these files in a samples/ directory)
SAMPLE_CONTRACTS = {
    "NDA Agreement": "samples/nda_sample.txt",
    "Service Contract": "samples/service_contract_sample.txt",
    "Employment Agreement": "samples/employment_sample.txt"
}

# Authentication Functions
def get_iam_token() -> str:
    """Fetch IAM token using API key from Streamlit secrets"""
    if "iam_token" in st.session_state and st.session_state.get("iam_token_expiry", 0) > time.time():
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

def call_granite_api(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call IBM Granite API with proper chat formatting and error handling."""
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
        "messages": [
            {
                "role": "system",
                "content": """You are a legal contract analysis AI. Follow these rules:
1. Respond with well-structured markdown
2. Use tables for comparative analysis
3. Highlight risks in **bold**
4. Format JSON responses with code blocks
5. Be precise and professional"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "project_id": st.secrets["GRANITE_INSTANCE_ID"],
        "parameters": {
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
    }

    try:
        with st.spinner("Analyzing with Granite model..."):
            response = requests.post(
                GRANITE_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_info = response.json().get("errors", [{}])[0]
                st.error(f"""API Error {response.status_code}:
- Message: {error_info.get('message', 'Unknown error')}
- Model: {model}
- Project ID: {st.secrets['GRANITE_INSTANCE_ID'][:6]}...
- Tip: Verify your project ID and model access""")
                return ""

            result = response.json().get("results", [{}])
            return result[0].get("generated_text", "")

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    
    return ""

def analyze_contract_with_granite(contract_text: str, custom_risks: List[str] = None) -> Dict:
    """Analyze contract text using Granite models."""
    risks = custom_risks if custom_risks else DEFAULT_RISK_CLAUSES
    standards = DEFAULT_STANDARD_CLAUSES
    
    prompt = f"""
    Analyze this contract and provide JSON output with:
    1. Risk score (1-10)
    2. Risky clauses (name, text, risk level, recommendation)
    3. Missing standard clauses
    4. Key obligations (party, obligation, deadline)
    5. Mitigation strategies
    6. Executive summary
    
    Contract Excerpt:
    {contract_text[:10000]}
    
    Risk Clauses to Check: {', '.join(risks)}
    Standard Clauses Expected: {', '.join(standards)}
    
    Format:
    ```json
    {{
        "risk_score": 0,
        "risky_clauses": [],
        "missing_clauses": [],
        "key_obligations": [],
        "mitigation_strategies": [],
        "summary": ""
    }}
    ```
    """
    
    response = call_granite_api(prompt)
    
    try:
        json_str = response.split('```json')[1].split('```')[0].strip()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Failed to parse response: {str(e)}")
        return {"error": str(e), "raw_response": response}

def generate_workflow(analysis_result: Dict, approvers: List[str]) -> Dict:
    """Generate approval workflow based on analysis results."""
    risk_score = analysis_result.get("risk_score", 0)
    steps = []
    
    if risk_score >= 7:
        steps = ["Legal Review", "Compliance Review", "Senior Management Approval"]
    elif risk_score >= 4:
        steps = ["Legal Review", "Department Head Approval"]
    else:
        steps = ["Legal Review"]
    
    if analysis_result.get("missing_clauses"):
        steps.append("Clause Addition Review")
    
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

def render_risk_analysis(analysis_result: Dict):
    """Render the risk analysis results."""
    st.subheader("📊 Risk Analysis Summary")
    
    risk_score = analysis_result.get("risk_score", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Risk Score", risk_score)
    with col2:
        st.metric("Risky Clauses", len(analysis_result.get("risky_clauses", [])))
    with col3:
        st.metric("Missing Clauses", len(analysis_result.get("missing_clauses", [])))
    
    with st.expander("🔍 Detailed Analysis", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Risky Clauses", "Missing Clauses", "Mitigation"])
        
        with tab1:
            if analysis_result.get("risky_clauses"):
                for clause in analysis_result["risky_clauses"]:
                    st.markdown(f"""
                    **{clause.get('clause_name', 'Unnamed Clause')}**  
                    *Risk Level: {clause.get('risk_level', 'Medium')}*  
                    {clause.get('text', '')}  
                    💡 **Recommendation:** {clause.get('recommendation', 'None provided')}
                    """)
                    st.divider()
            else:
                st.success("✅ No risky clauses identified")
        
        with tab2:
            if analysis_result.get("missing_clauses"):
                st.write("Missing standard clauses:")
                for clause in analysis_result["missing_clauses"]:
                    st.markdown(f"- {clause}")
            else:
                st.success("✅ All standard clauses present")
        
        with tab3:
            if analysis_result.get("mitigation_strategies"):
                st.write("Recommended mitigation strategies:")
                for strategy in analysis_result["mitigation_strategies"]:
                    st.markdown(f"- {strategy}")
            else:
                st.info("No specific mitigation strategies recommended")

def main():
    # Initialize session state
    if "iam_token" not in st.session_state:
        st.session_state.iam_token = ""
        st.session_state.iam_token_expiry = 0
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.title("⚖️ Contract Analyzer")
        
        uploaded_file = st.file_uploader(
            "Upload Contract",
            type=SUPPORTED_FILE_TYPES,
            help="Supported formats: PDF, Word, Text"
        )
        
        st.markdown("---")
        st.markdown("**Sample Contracts**")
        sample_contract = st.selectbox(
            "Choose sample",
            options=list(SAMPLE_CONTRACTS.keys()),
            label_visibility="collapsed"
        )
        
        if st.button("Load Sample"):
            try:
                with open(SAMPLE_CONTRACTS[sample_contract], "r") as f:
                    st.session_state.contract_text = f.read()
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
        
        st.markdown("---")
        with st.expander("Advanced Options"):
            custom_risks = st.text_area(
                "Custom Risk Clauses",
                value="\n".join(DEFAULT_RISK_CLAUSES),
                help="One clause per line"
            )
            st.session_state.custom_risks = [r.strip() for r in custom_risks.split("\n") if r.strip()]
            
            approvers = st.text_input(
                "Approvers (comma separated)",
                value="Legal, Compliance, Manager"
            )
            st.session_state.approvers = [a.strip() for a in approvers.split(",") if a.strip()]

    # Main content
    st.title("AI Contract Risk Analyzer")
    
    # Check for required secrets
    if not all(key in st.secrets for key in ["IAM_API_KEY", "GRANITE_INSTANCE_ID"]):
        st.error("Missing required secrets (IAM_API_KEY, GRANITE_INSTANCE_ID)")
        return
    
    # Process uploaded file
    if uploaded_file and not st.session_state.get("file_processed"):
        with st.spinner("Processing document..."):
            try:
                st.session_state.contract_text = clean_text(extract_text_from_file(uploaded_file))
                st.session_state.file_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Display analysis
    if st.session_state.get("file_processed") and st.session_state.get("contract_text"):
        contract_text = st.session_state.contract_text
        
        tab1, tab2, tab3 = st.tabs(["Analysis", "Workflow", "Q&A"])
        
        with tab1:
            if "analysis_result" not in st.session_state:
                with st.spinner("Analyzing contract..."):
                    st.session_state.analysis_result = analyze_contract_with_granite(
                        contract_text,
                        st.session_state.get("custom_risks")
                    )
            
            if "analysis_result" in st.session_state:
                render_risk_analysis(st.session_state.analysis_result)
                
                st.download_button(
                    "Download Analysis",
                    json.dumps(st.session_state.analysis_result, indent=2),
                    "contract_analysis.json"
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
                    for step in st.session_state.workflow["steps"]:
                        due_date = datetime.strptime(step["due_date"], "%Y-%m-%d")
                        days_left = (due_date - datetime.now()).days
                        
                        cols = st.columns([1, 3, 1])
                        cols[0].write(f"**{step['step']}**")
                        cols[1].progress(min(100, 100 - days_left), f"Due in {days_left} days")
                        cols[2].write(f"👤 {step['approver']}")
                        st.caption(f"Status: {step['status']} • {step['required']}")
                        st.divider()
        
        with tab3:
            st.subheader("💬 Contract Q&A")
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
    
    else:
        st.markdown("""
        ## Welcome to the AI Contract Risk Analyzer
        
        1. Upload a contract document
        2. Or select a sample contract
        3. View risk analysis and generate workflows
        
        *Note: This tool uses IBM watsonx.ai for analysis*
        """)

if __name__ == "__main__":
    main()
