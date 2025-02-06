from django.http import JsonResponse
import fitz, json,requests
from pptx import Presentation
from PIL import Image
from docx import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from typing import List
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from fastapi.responses import HTMLResponse
from fastapi import Request
import re
from fastapi import  File, UploadFile, Form
import requests
import subprocess
import time

templates = Jinja2Templates(directory="templates")

# Databricks Configuration (Replace with actual values)
os.environ["DATABRICKS_HOST"] = "https://adb-335959.13.azuredatabricks.net/"
os.environ["DATABRICKS_TOKEN"] = "**********"

DATABRICKS_HOST = "https://adb-335959.13.azuredatabricks.net/"
DATABRICKS_TOKEN = "******************"
mlflow.set_tracking_uri("databricks")

def get_github_repos(user_pat):
    print(user_pat)
    url = "https://api.github.com/user/repos"
    
    headers = {
        "Authorization": f"token {user_pat}",
        "Accept": "application/vnd.github.v3+json"
    }    
    response = requests.get(url, headers=headers)
    print(response)    
    if response.status_code == 200:
        return JsonResponse({'status': 'success', 'repos': response.json()})
    else:
        return JsonResponse({'status': 'error', 'message': response.json()})
    


def extract_text_from_doc(message=None):
    docx_file_path = "media/model.docx"   
    try:
        document = Document(docx_file_path)
        paragraphs = document.paragraphs
        matching_paragraphs = []
        from .prompt import word_prompt
        import re
        url_pattern = re.compile(r'(https?://\S+)')
        for paragraph in paragraphs:
            if message and any(keyword in paragraph.text.lower() for keyword in word_prompt if keyword in message.lower()):
                text = paragraph.text
                text_with_links = url_pattern.sub(r'<a href="\1">\1</a>', text)
                matching_paragraphs.append(text_with_links)

        if not matching_paragraphs:
            return "Sorry, I couldn't find relevant information in the Model."
        return " ".join(matching_paragraphs)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    


from pydantic import BaseModel

class ChatbotRequest(BaseModel):
    query: str



def get_experiments_list():
    """
    Processes query to list available experiments.

    Args:
        query: User query string.

    Returns:
        Dictionary containing a list of experiments or an error message.
    """

    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        experiment_metadata = []
        for i, exp in enumerate(experiments):
            if i >= 5:
                break  # Limit loop to 5 iterations
            experiment_metadata.append({
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "artifact_location": exp.artifact_location,
            })
        # Return structured response

        # experiment_metadata = [
        #     {
        #         "name": exp.name,
        #         "experiment_id": exp.experiment_id,
        #         "artifact_location": exp.artifact_location,
        #     }
        #     for exp in experiments
        # ]

        return {"status": "success", "experiments": experiment_metadata}

    except Exception as e:
      return {"status": "error", "detail": str(e)}

def get_model_list():
    """
    Processes query to list available models.

    Args:
        query: User query string.

    Returns:
        Dictionary containing a list of models or an error message.
    """

    try:
      # Retrieve all registered models
      client = mlflow.tracking.MlflowClient()
      models = client.search_registered_models()
      model_metadata = [
          {
              "name": model.name,
              "latest_version": model.latest_versions[0].version,
              "creation_timestamp": model.creation_timestamp,
          }
          for model in models
      ]
      return {"models": model_metadata}

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error retrieving models: {str(e)}")



def model_registration(model_name, model_description, repo_url):
        timestamp = int(time.time())
        #repo_dir = f"/tmp/mossaicai_demo_{timestamp}"
        
        repo_dir = "/home/ajitm/fast_chat/app/"

        try:
            # Clone the repository using SSH
            #subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

            # Change directory to the cloned repository
            os.chdir(repo_dir)

            # Assuming the model training script is `train.py` in the repo
            train_script = "sample_registration.py"
            #train_script = "/home/ajitm/fast_chat/app/sample_registration.py"
            #if not os.path.exists(train_script):
            #    raise HTTPException(
            #        status_code=404, 
            #        detail=f"Training script '{train_script}' not found in the repository."
            #    )
            
            # Use MLflow to register the model
            client = mlflow.tracking.MlflowClient()
            # Run the training script with model_name as argument
            result = subprocess.run(
                ["python", train_script, model_name],
                capture_output=True, 
                text=True, 
                check=True
            )

            registered_model = client.get_registered_model(model_name)            
            #print("Script output:", result.stdout)

            registered_model = client.get_registered_model(model_name)
    
            # Construct the Databricks model link
            #workspace_url = f"{DATABRICKS_HOST}.cloud.databricks.com"
            model_url = f"{DATABRICKS_HOST}ml/models/{model_name}/versions/{registered_model.latest_versions[-1].version}"
            print(model_url)
                        
            accuracy=re.search(r"accuracy:\s*([\d\.]+)", result.stdout).group(1)    

            # Extract run_url
            #run_url = re.search(r"View run (https?://[^\s]+)", result.stdout).group(1)
                     
            # Extract experiment_url
            #experiment_url = re.search(r"View experiment at: (https?://[^\s]+)", result.stdout).group(1)
      
            return {
                "message": f"Model '{model_name}' registered successfully with the model description '{model_description}'.<br><br>",
                "view_model_link": f"<a href='{model_url}'>View Model</a><br><br>"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving model metadata: {str(e)}")



def get_model_metadata():
  """
  Processes query to get specific model metadata.

  Args:
      query: User query string.

  Returns:
      Dictionary containing model metadata or an error message.
  """

  if "get model metadata" in query:
    # Use regex to extract the model name (function not shown for brevity)
    model_name_match = re.search(r"model_name=([^\s,]+)", query)
    if model_name_match:
      model_name = model_name_match.group(1)  # Extracted model name
      print(f"Model Name: {model_name}")
    else:
      print("Model name not found in the query.")
    if not model_name:
      raise HTTPException(status_code=400, detail="Model name must be provided for this query.")

    try:
      # Retrieve model metadata by name
      client = mlflow.tracking.MlflowClient()
      registered_models = client.search_registered_models(f"name='{model_name}'")

      if not registered_models:
        raise HTTPException(status_code=404, detail=f"No registered model found with name: {model_name}")

      model_metadata = []
      for model in registered_models:
        for version in model.latest_versions:
          model_metadata.append({
              "model_name": model.name,
              "version": version.version,
              "stage": version.current_stage,
              "status": version.status,
              "source": version.source,
              "run_id": version.run_id,
          })
      return {"model_metadata": model_metadata}

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error retrieving experiments: {str(e)}")
  else:
    return None  # Not a get model metadata request


def get_model_metadata_wrapper(request):
  """
  Wrapper function to call appropriate handler based on query.

  Args:
      request: Chatbot request object.

  Returns:
      Dictionary containing response data or an error message.
  """

  query = request.query
  # Call respective function based on query intent
  response = list_models(query) or list_experiments(query) or get_model_metadata(query)
  if response:
    return response
  else:
    # No matching intent found
    raise HTTPException(status_code=400, detail=f"Error retrieving metadata: {str(e)}")


