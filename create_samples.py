import os

# Create samples directory if it doesn't exist
os.makedirs("samples", exist_ok=True)

# Sample content
samples = {
    "nda_sample.txt": """NON-DISCLOSURE AGREEMENT\n\nBetween [Company] and [Recipient].\n\n1. CONFIDENTIAL INFORMATION\nAll business information shared.\n2. TERM\n2 years validity.""",
    "service_contract_sample.txt": """SERVICE AGREEMENT\n\nBetween [Provider] and [Client].\n\n1. SERVICES\n[Describe services].\n2. PAYMENT\n$[amount] per hour.""",
    "employment_sample.txt": """EMPLOYMENT CONTRACT\n\nBetween [Company] and [Employee].\n\n1. POSITION\n[Job Title].\n2. COMPENSATION\n$[amount] per year."""
}

# Create files
for filename, content in samples.items():
    with open(f"samples/{filename}", "w") as f:
        f.write(content)

print("Sample contract files created successfully in the 'samples' folder!")
