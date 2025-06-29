# AI Contract Analyzer

An IBM Granite-powered contract risk assessment tool.

## Setup
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set up secrets (see below)

## Configuration
Create `.streamlit/secrets.toml` with:
```toml
IAM_API_KEY = "your_ibm_cloud_key"
GRANITE_INSTANCE_ID = "your_project_id"
