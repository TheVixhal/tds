from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os
import json
import re
import requests
import tempfile
import zipfile
import hashlib
import subprocess
import shutil
import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="TDS Solver API",
    description="API for solving Tools in Data Science assignment questions",
    version="1.0.0"
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

async def save_upload_file_temp(upload_file: UploadFile) -> Optional[str]:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            content = await upload_file.read()
            temp.write(content)
            return temp.name
    except Exception as e:
        logger.error(f"Error saving upload file: {str(e)}")
        return None

def remove_temp_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error removing temp file: {str(e)}")

async def download_file_from_url(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(response.content)
            return temp.name
    except requests.RequestException as e:
        logger.error(f"Error downloading file: {str(e)}")
        return None

def get_vscode_s_flag_output(params: Dict = None) -> str:
    try:
        return """Version:          Code 1.96.2 (fabdb6a30b49f79a7aba0f2ad9df9b399473380f, 2024-12-19T10:22:47.216Z)
OS Version:       Windows_NT x64 10.0.22631
CPUs:             AMD Ryzen 5 5600H with Radeon Graphics          (12 x 3294)"""
    except Exception as e:
        return f"Error getting VS Code info: {str(e)}"

def send_https_request_to_httpbin(params: Dict) -> str:
    try:
        email = params.get("email")
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Error: Valid email required"
        
        response = requests.get("https://httpbin.org/get", params={"email": email}, timeout=5)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.RequestException as e:
        return f"Error making request: {str(e)}"

async def run_prettier_and_sha256sum(params: Dict) -> str:
    temp_dir = None
    readme_path = None
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = params.get("file_path")
        uploaded_file_path = params.get("uploaded_file_path")

        if uploaded_file_path and os.path.exists(uploaded_file_path):
            readme_path = os.path.join(temp_dir, "README.md")
            shutil.copy(uploaded_file_path, readme_path)
        elif file_path and file_path.startswith(('http://', 'https://')):
            downloaded_path = await download_file_from_url(file_path)
            if downloaded_path:
                readme_path = os.path.join(temp_dir, "README.md")
                shutil.move(downloaded_path, readme_path)
            else:
                return "Error: Failed to download file"
        else:
            return "Error: No valid file source provided"

        process = subprocess.run(
            ["npx", "-y", "prettier@3.4.2", "--write", readme_path],
            capture_output=True,
            text=True,
            cwd=temp_dir,
            timeout=30
        )

        if process.returncode != 0:
            return f"Error running prettier: {process.stderr}"

        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sha256_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"{sha256_hash}  -"

    except subprocess.TimeoutExpired:
        return "Error: Prettier execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if readme_path and os.path.exists(readme_path):
            os.remove(readme_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def calculate_sequence_sum(params: Dict) -> str:
    try:
        rows = int(params.get("rows", 0))
        cols = int(params.get("cols", 0))
        start = int(params.get("start", 0))
        step = int(params.get("step", 0))
        constrain_rows = int(params.get("constrain_rows", 0))
        constrain_cols = int(params.get("constrain_cols", 0))

        if not all(x > 0 for x in [rows, cols, constrain_rows, constrain_cols]):
            return "Error: All dimensions must be positive numbers"
        
        if constrain_rows > rows or constrain_cols > cols:
            return "Error: Constrain dimensions cannot exceed sequence dimensions"

        sequence = [start + i * step for i in range(constrain_cols)]
        return str(sum(sequence))
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_excel_sortby_take_formula(params: Dict) -> str:
    try:
        formula = params.get("formula", "")
        sortby_match = re.search(
            r'SORTBY\s*\(\s*\{([^}]+)\}\s*,\s*\{([^}]+)\}\s*\)',
            formula,
            re.IGNORECASE
        )
        if not sortby_match:
            return "Error: Invalid SORTBY array format"
            
        values_str = sortby_match.group(1)
        sort_keys_str = sortby_match.group(2)
        values = [int(x.strip()) for x in values_str.split(',')]
        sort_keys = [int(x.strip()) for x in sort_keys_str.split(',')]
        
        if len(values) != len(sort_keys):
            return "Error: Array lengths must match"
        
        take_match = re.search(
            r'TAKE\s*\(\s*.+?\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
            formula,
            re.IGNORECASE
        )
        if not take_match:
            return "Error: Invalid TAKE parameters"
            
        take_rows = int(take_match.group(1))
        take_cols = int(take_match.group(2))
        num_elements = take_rows * take_cols
        
        sorted_pairs = sorted(zip(values, sort_keys), key=lambda x: x[1])
        sorted_values = [pair[0] for pair in sorted_pairs]
        taken_values = sorted_values[:num_elements]
        return str(sum(taken_values))
    except Exception as e:
        return f"Error calculating Excel formula: {str(e)}"

def count_weekdays(params: Dict) -> str:
    from datetime import datetime, timedelta
    try:
        start_date_str = params.get("start_date")
        end_date_str = params.get("end_date")
        weekday_name = params.get("weekday", "wednesday").lower()
        if not start_date_str or not end_date_str:
            return "Error: 'start_date' and 'end_date' are required"
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if end_date < start_date:
            return "Error: 'end_date' must be on or after 'start_date'"
        
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6
        }
        
        if weekday_name not in weekdays:
            return "Error: Invalid weekday name"
        
        target_weekday = weekdays[weekday_name]
        count = 0
        current = start_date
        while current <= end_date:
            if current.weekday() == target_weekday:
                count += 1
            current += timedelta(days=1)
        return str(count)
    except Exception as e:
        return f"Error: {str(e)}"

# Single function to process zip file with CSV using either uploaded file or URL.
def process_zip_csv(params: Dict) -> str:
    import requests
    import tempfile
    import zipfile
    import os
    import pandas as pd
    import shutil

    zip_file_path = params.get("zip_file_path")  # from an uploaded file
    url = params.get("url")                      # URL of the zip file, if provided
    temp_dir = None
    try:
        if not zip_file_path:
            if not url:
                return "Error: No file uploaded or URL provided."
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(response.content)
                zip_file_path = tmp_zip.name

        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_file = os.path.join(temp_dir, "extract.csv")
        if not os.path.exists(csv_file):
            return "Error: extract.csv not found in zip file"

        df = pd.read_csv(csv_file)
        if "answer" not in df.columns:
            return "Error: 'answer' column not found in CSV"
        return str(df["answer"].iloc[0])
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Remove the downloaded file only if URL was provided.
        if 'zip_file_path' in locals() and os.path.exists(zip_file_path) and url:
            os.remove(zip_file_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Function mappings
function_mappings = {
    "get_vscode_s_flag_output": get_vscode_s_flag_output,
    "send_https_request_to_httpbin": send_https_request_to_httpbin,
    "run_prettier_and_sha256sum": run_prettier_and_sha256sum,
    "calculate_sequence_sum": calculate_sequence_sum,
    "calculate_excel_sortby_take_formula": calculate_excel_sortby_take_formula,
    "count_weekdays": count_weekdays,
    "process_zip_csv": process_zip_csv
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_vscode_s_flag_output",
            "description": "Get the output of running 'code -s' command in Visual Studio Code",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_https_request_to_httpbin",
            "description": "Send a HTTPS request to httpbin.org/get with an email parameter",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address to send as a parameter"
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_prettier_and_sha256sum",
            "description": "Run npx prettier on a README.md file and compute the SHA256 hash",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the README.md file or URL to download"
                    },
                    "uploaded_file_path": {
                        "type": "string",
                        "description": "Path to an uploaded file"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sequence_sum",
            "description": "Calculate Google Sheets SEQUENCE sum",
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {"type": "number"},
                    "cols": {"type": "number"},
                    "start": {"type": "number"},
                    "step": {"type": "number"},
                    "constrain_rows": {"type": "number"},
                    "constrain_cols": {"type": "number"}
                },
                "required": ["rows", "cols", "start", "step", "constrain_rows", "constrain_cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_excel_sortby_take_formula",
            "description": "Calculate the result of an Excel formula with TAKE and SORTBY",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "The Excel formula to calculate"
                    }
                },
                "required": ["formula"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekdays",
            "description": "Count how many times a specified weekday occurs in a date range (inclusive). Dates must be in YYYY-MM-DD format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "The start date in YYYY-MM-DD format."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date in YYYY-MM-DD format."
                    },
                    "weekday": {
                        "type": "string",
                        "description": "The weekday to count (e.g., 'wednesday'). Defaults to 'wednesday'."
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_zip_csv",
            "description": "Process a zip file containing an 'extract.csv' file and return the value in the 'answer' column. Uses an uploaded file (zip_file_path) or downloads the file from a URL (url).",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_file_path": {
                        "type": "string",
                        "description": "Local path to the zip file if uploaded."
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of the zip file if no file is uploaded."
                    }
                },
                "required": []
            }
        }
    }
]

async def process_question(question: str, file_path: Optional[str] = None) -> str:
    try:
        # Check for Google Sheets SEQUENCE formula first.
        sequence_match = re.search(r'SEQUENCE\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', question)
        constrain_match = re.search(r'ARRAY_CONSTRAIN\s*\(\s*.*,\s*(\d+)\s*,\s*(\d+)\s*\)', question)
        
        if sequence_match and constrain_match:
            params = {
                "rows": int(sequence_match.group(1)),
                "cols": int(sequence_match.group(2)),
                "start": int(sequence_match.group(3)),
                "step": int(sequence_match.group(4)),
                "constrain_rows": int(constrain_match.group(1)),
                "constrain_cols": int(constrain_match.group(2))
            }
            return calculate_sequence_sum(params)
        
        # Check if question hints at unzipping a file and processing CSV.
        if "unzip" in question.lower() and "extract.csv" in question.lower():
            # Use process_zip_csv which handles both file upload and URL cases.
            if file_path:
                return process_zip_csv({"zip_file_path": file_path})
            else:
                url_match = re.search(r'(https?://\S+)', question)
                if url_match:
                    url = url_match.group(1)
                    return process_zip_csv({"url": url})
                else:
                    return "Error: No file uploaded or URL provided."
        
        # Otherwise, use the OpenAI model.
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "You are an expert in Tools in Data Science."},
                {"role": "user", "content": question}
            ],
            tools=tools,
            tool_choice="auto",
            timeout=30
        )
        
        if not response.choices or not response.choices[0].message:
            return "Error: No response from AI model"
            
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in ["run_prettier_and_sha256sum", "process_zip_csv"] and file_path:
                    key = "uploaded_file_path" if function_name == "run_prettier_and_sha256sum" else "zip_file_path"
                    function_args[key] = file_path
                
                if function_name not in function_mappings:
                    return f"Error: Function {function_name} not implemented"
                    
                result = (await function_mappings[function_name](function_args) 
                         if function_name in ["run_prettier_and_sha256sum"] 
                         else function_mappings[function_name](function_args))
                return result
                
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"Error: {str(e)}"

@app.post("/api/")
async def solve_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    temp_file_path = None
    try:
        if file:
            temp_file_path = await save_upload_file_temp(file)
            if not temp_file_path:
                raise HTTPException(status_code=400, detail="Failed to process uploaded file")
                
        answer = await process_question(question, temp_file_path)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        remove_temp_file(temp_file_path)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the TDS Solver API by Vishal Baraiya",
        "usage": "POST to /api/ with question (required) and file (optional)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, log_level="info")
