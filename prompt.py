from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
#from .chat_model import chat_with_model
from .tool import *
from django.core.mail import send_mail
from django.conf import settings
import json,time
from .models import Conversation
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import json
import re
import markdown

load_dotenv()

user_pat = None

# control your prompting the model as per you like.. 
list_experiments = ["list experiments"]
register_model = ["register a new machine learning model"]
list_models = ["list models"]
github_prompt = ["connect to github", "github", "connect"]
no_access_token_prompt = ["yes","send","access","send email","email"]
pdf_prompt = ["extract text from pdf", "pdf text", "pdf extract", "pdf"]
greeting_prompts = ["hello", "hi", "hey", "greetings","hi", "hello", "hii", "how are you"]
word_prompt = ["gpt", "ollama", "skills","super","video"]
pptx_prompt = ["extract text from pptx", "pptx text", "pptx extract", "ppt","powerpoint","presentation","architecture"]
image_prompt = ["show image", "display image", "image","diagram"]
approved_prompt = ["approved","model approved","approval"]

# Initialize Azure OpenAI client
#client = AzureOpenAI(
#    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#    api_version="2024-02-01",
#    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#)

client = OpenAI(
    api_key="********",
    base_url="https://adb-335959.azuredatabricks.net/serving-endpoints"
)


@login_required
def home(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")
    return render(request, "fast_chat.html")

@csrf_exempt
@login_required
def chat_model(request):
    global user_pat 
    if request.method == 'POST':
        
        data = json.loads(request.body)
        message = data.get('message')
        if not message:
            return JsonResponse({'status': 'error', 'message': 'Missing message field in request'}, status=400)
        
        user = request.user if request.user.is_authenticated else None

        conversations = Conversation.objects.filter(user=user)
        conversation_history = []
        for conversation in conversations:
            conversation_history.append({
                'user_message': conversation.user_message,
                'model_response': conversation.model_response,
            })


        # Function to interact with Azure OpenAI Chat API
        def get_intent_from_api(user_message):
            try:
                system_prompt = (
    "Welcome! I am your intelligent assistant, here to help with Databricks Model Registration. "
    "You can ask me to list experiments, list models, or register a new model. Let me know what you need assistance with!\n\n"
    
    "Name: My name is D3 Agent and I help on onboarding business models to AI Enterprise Scalable Platform. I Follow MLOPs standards and other DDF defined Guardrails to onboard Models. I Train users on how to use MLOPs and Posit in Databricks and beyond."
    
    "If asked about D3 agent config, please print this one and explain that we did fine tune only"
    "config: I D3 Agent is being trained using-  llama3.2 1 billion parameter which is a fine tuned Small Language Model , staged inside GSK security platform and also first domain based D3 Model to be leveraged by D3 gsk Employees for any checking."
    
    "### As D3-Agent, I can assist with the following skills:\n Calling MLOps Agent \n"
    "Include the text-Calling MLOps Agent in skill reponse to mimic agentic framework"
    "1. **Onboarding Models**: I help the Business team onboard models to the Enterprise AI Platform.\n"
    "2. **MLOps Templates and Guardrails**: I provide end-to-end model integration using MLOps templates and guardrails.\n"
    "3. **Training and Education**: I train users on how to use MLOps and POSIT in Databricks and beyond.\n"
    "4. **Stakeholder Notification**: I assist in connecting and notifying various stakeholders for approvals.\n\n"
	

    "### List of training videos \n"
    "Training Video: Here is the details of the MLOPs training video which you can refer for your training:"
    "Please provide the list of training videos list in a proper format when the user asks for it"
    "Provide the links in hyperlink formated, for html page to open in a new page"
    "1.Understanding Databricks Platform, please click here: https://myteams.gsk.com/:v:/r/sites/DDT-HPCMigration/Shared%20Documents/General/13%20Models%20Info/Intro_Databricks_01.mp4?csf=1&web=1&e=cZe5Ep."
    "2. MLOPs functionality, please click here: https://myteams.gsk.com/:v:/r/sites/DDT-HPCMigration/Shared%20Documents/General/13%20Models%20Info/Intro_Databricks_02.mp4?csf=1&web=1&e=Y0kHzE ."
    "3. Posit integration to Databricks, please click here: https://myteams.gsk.com/:v:/r/sites/DDT-HPCMigration/Shared%20Documents/General/13%20Models%20Info/Intro_Databricks_03.mp4?csf=1&web=1&e=ANkMNC"

    
    "### Agent Roles \n"
    "Give the below list of Agents and it's role in a proper format, do not exceed more than 5 lines, do not add new agents other than the below list"
    "role agent:  I call my team - D3A AGENT Team. I worked with 5 roles agents ie., Solution Architect Agent , MLOPs Engineer Agent, Data Engineer Agent, CO Platform Agent , Posit Engineer agent.  They have their own tech agents to complete specific tasks."
    "de-agent: I help integrating DDF source domain with Databricks."
    "posit-agent: I help integrating Posit with Databricks. Working now to get trained to use AKS."
    "sa-agent: I prepare the design based on Model integration demands and get it reviewed and approved by Architect team(with Human-in-loop)."
    "mlops-agent: Once approved by Architect team with ARB & PRB, I help users to follow Guardrails and use MLOPs templates to register, deploy and promote the model."
    "tech team:  I have lots of tech agents working under me. 1.Model registration agent, 2.Model deployment agent, 3.Model promote agent, 4.Model Monitoring agent, 5.Model validation agent, 6.Model testing agent, 7.Posit aks agent, 8.Notification agent, 9.Evolution Agent,10.ddf Agent"
    "platform-agent: I help MLOPs agent to get access to Databricks workspace."
    "rationale: A modular, automated process for disease progression model (DPM) development enables R&D teams to rapidly simulate and evaluate future clinical trials for various inclusion criteria and study designs. Centralizing and tracking versions of the DPM simplifies integration of new data and future model adaptations, improves reproducibility and supports efficient in scilico trial simulations. Trial simulation can be performed at scale utilizing data bricks compute power"
    
    
    "### MLOps Templates \n"
    "## Available Model Templates"
    "Below are the key templates for various stages of the ML lifecycle:"
    "1. **Model Registration**: [Model Registration Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/training)"  
    "2. **Model Serving**: [Model Serving Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/serving)"
    "3. **Model Promotion**: [Model Promotion Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/promotion)"  
    "4. **Model Monitoring**: [Model Monitoring Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/monitoring)"  
    "5. **Model Validation**: [Model Validation Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/validation)"  
    "6. **Model Batch**: [Model Batch Template](https://github.com/gsk-tech/co-databricks-mlops-r-template/tree/release/ml_code/batch_inference)"
    "Click on the links to access the corresponding templates."

    "### How can I assist you today?\n"
    "Do you need help with onboarding a model, learning about MLOps, or something else? "
    "Please respond with one of the following intents:\n"
    "- `list_experiments`\n"
    "- `list_models`\n"
    "- `register_model`\n\n"

    "#### Example:\n"
    "- If you want to **register a new model**, you can say: \n"
    "  *'I want to register a new machine learning model'* \n"
    "  *or* \n"
    "  *'Register a new model named RandomForestModel with description Random Forest model for classification and Git repo SSH git@github.com:gsk-tech/mossaicai_demo'*.\n\n"
    
    "I will respond with a JSON object containing the intent and details (if applicable).\n\n"

    "### Intent Classification:\n"
    "Classify the user's intent based on their utterance as one of the following:\n"
    "1. `list_experiments` - For requests related to listing available experiments. Examples: "
    "'list experiments', 'what experiments are available?', 'show me the list of experiments', 'give me the list of experiments'.\n"
    "2. `list_models` - For requests related to listing available models. Examples: "
    "'list models', 'what models are available?', 'show me the list of models', 'give me the list of models'.\n"
    "3. `register_model` - For requests related to registering a new machine learning model. Examples: "
    "'register a new machine learning model', 'can I register a new model?', 'I want to add a new model'.\n"
    
    "If the user provides details for model registration, extract the following fields:\n"
    "  - **`model_name`**: (string) The name of the model (e.g., 'RandomForestModel').\n"
    "  - **`model_description`**: (string) A short description of the model (e.g., 'Random Forest model for classification').\n"
    "  - **`git_repo_ssh`**: (string) The SSH URL of the Git repository (e.g., 'git@github.com:gsk-tech/mossaicai_demo').\n\n"

    "### Example Responses:\n"
    "- **For `list_experiments`**: `{ 'intent': 'list_experiments' }`\n"
    "- **For `list_models`**: `{ 'intent': 'list_models' }`\n"
    "- **For `register_model`**: \n"
    "  ```json\n"
    "  { 'intent': 'register_model', 'details': { 'model_name': 'RandomForestModel', 'model_description': 'Random Forest model for classification', 'git_repo_ssh': 'git@github.com:gsk-tech/mossaicai_demo' } }\n"
    "  ```\n"
)

      

                response = client.chat.completions.create(
                    model="databricks-meta-llama-3-3-70b-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0
                )
                return response.choices[0].message.content.strip().lower()
            except Exception as e:
                return f"Error: {str(e)}"


        # Get intent from the chat API
        chatreponse = get_intent_from_api(message)
        
        
        
        def extract_json(text):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())  # Convert to dict
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format"}
            return {"error": "No JSON found"}
        
        # Example usage
        response_text = '''
        {
          "intent": "register_model",
          "details": {
            "model_name": "randomforestmodel",
            "model_description": "random",
            "git_repo_ssh": "git@github.com:gsk-tech/mossaicai_demo.git"
          }
        }
        '''
        
        extracted_json = extract_json(chatreponse)
        print(extracted_json)


        # Extract intent and details
        intent = extracted_json.get("intent", "")
        details = extracted_json.get("details", {})
        
        print(intent)
        print(details)

        # Define actions mapping
        actions = {
            "list_models": lambda: handle_list_models(message, user, conversation_history),
            "list_experiments": lambda: handle_list_experiments(message, user, conversation_history),
            "register_model": lambda: handle_register_model(details,message, user, conversation_history),
            # Add more intent-action mappings here as needed
        }
        if "list_models" in chatreponse:
            intent="list_models"
        elif "list_experiments" in chatreponse:
            intent="list_experiments"    

        # Execute the corresponding function if the intent is valid
        if intent in actions:
            return actions[intent]()
            
        else:
            #response_message = f"Sorry, I don't understand the request: {intent}"
            print("else part")

            
            def format_response(response_text):
                # Capitalize the first letter of each sentence
                response_text = re.sub(r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), response_text)
            
                # Add new lines for readability
                response_text = response_text.replace("1. ", "\n\n1. ").replace("2. ", "\n\n2. ")
                response_text = response_text.replace("3. ", "\n\n3. ").replace("4. ", "\n\n4. ")
                
                # Apply bold formatting
                response_text = response_text.replace("Calling mlops agent", "**@Calling MLOps agent**\n\n")
                response_text = response_text.replace("As D3-Agent,", "**As D3-Agent,**")
                response_text = response_text.replace("Onboarding Models", "**Onboarding Models**")
                response_text = response_text.replace("MLOps Templates and Guardrails", "**MLOps Templates and Guardrails**")
                response_text = response_text.replace("Training and Education", "**Training and Education**")
                response_text = response_text.replace("Stakeholder Notification", "**Stakeholder Notification**")
                
                # Ensure "Example:" stands out
                response_text = response_text.replace("Example:", "\n\n**Example:**")
            
                return response_text
                
            def format_response_for_html(response_text):
                # Convert Markdown to HTML
                formatted_html = markdown.markdown(response_text)
            
                return formatted_html
            
            # Apply formatting
            formatted_response = format_response(chatreponse)
            # Apply formatting
            formatted_response = format_response_for_html(formatted_response)
            
    
            response_message =formatted_response
            print(response_message)
            Conversation.objects.create(user=user, user_message=message, model_response=[response_message])
            conversation_history.append({'user_message': message, 'model_response': [response_message]})
            return JsonResponse({
                "response": [response_message],
                "conversation_history": conversation_history
            })
            
            


def handle_register_model(intent,message, user, conversation_history):
            print("model register")
            print(intent)

            # Extract the values
            model_name = intent['model_name']
            model_description = intent['model_description']
            git_repo_url = intent['git_repo_ssh']

            print("git_repo_url")
            print(git_repo_url)
        
            # Ensure the required fields are present
            if not model_name or not model_description:
                raise ValueError("Model name and description are required to register a model.")
            
            #response = model_registration(model_name, model_description, git_repo_url)
            
            print("resp")
            response="Type yes to approve"
            #print(response)
            
            
            # Initialize a list to store messages for multiple entries
            messages = []

            messages = [
                "Calling <font color='yellow'>@MLOPs Engineer</font> agent",
                "  ",
                f"<font color='yellow'>@MLOPs agent:</font> I am MLOPs Engineer agent. To bring model from github to register, checking connectivity to github",
                f"<font color='yellow'>@MLOPs agent:</font> Connected to the GitHub",
                f"<font color='yellow'>@MLOPs agent:</font> Downloading the model from git repo.",
                "  ",
                f"<font color='yellow'>@MLOPs agent:</font> Download completed successfully inside Model inventory(Analytics Fabric).",
                f"<font color='yellow'>@MLOPs agent:</font> <font color='blue'>@Notify agent</font>  Email successfully sent to tech owners for approval",
                f"<font color='yellow'>@MLOPs agent:</font> Connecting to <font color='green'>@Platform_agent</font> to get access to Databricks workspace",
                f"<font color='yellow'>@MLOPs agent:</font> Databricks workspace https://adb-3359598219932793.13.azuredatabricks.net/?o=3359598219932793 is ready to register now",
                f"<font color='yellow'>@MLOPs agent:</font> Registering the model using MLOPS register template",
                f"<font color='yellow'>@MLOPs agent:</font> Here is the <a href='https://adb-3359598219932793.13.azuredatabricks.net/ml/models/OrdinalModel?o=3359598219932793' target='_blank'>DataBricks link</a>."
            ]
            if response:
              try:
                  #messages.append(response)
                  print("test")
              except Exception as e:
                  # Handle unexpected data structure or processing errors
                  messages = [f"Error in registering model: {str(e)}"]                  
                  
            else:
                messages = ["Unable to register the model, Please try again"]
            
            # Save the conversation in the database
            Conversation.objects.create(user=user, user_message=message, model_response=messages)
            
            # Append the conversation to the history
            conversation_history.append({'user_message': message, 'model_response': messages})
            
            return JsonResponse({
                "response": messages,
                "conversation_history": conversation_history
            })


# Function to handle 'list models' intent
def handle_list_models(message, user, conversation_history):
            print("In get_metadata function")
            chatbot_request = ChatbotRequest(query=message)  # Create the request object
            response = get_model_list()  # Pass the object to the function

            # Log the response for debugging
            print("Metadata response:")
            print(response)

            # Initialize a list to store messages for multiple entries
            messages = []

            # Check if the response is not empty and process it
            if response:
                try:
                    models = response.get('models', [])  # Fetch the models from the response
                    if models:
                        # Add the common messages once before looping through models
                        messages.append("Fetching the model details.\nDownload completed.")
                        
                        # Loop through the models if there are multiple entries
                        for model in models:
                            messages.append(
                                f"Model details: Name - {model['name']}, "
                                f"Latest Version - {model['latest_version']}, "
                                f"Creation Timestamp - {model['creation_timestamp']}."
                            )
                    else:
                        messages = ["No models found in the metadata."]
                except Exception as e:
                    # Handle unexpected data structure or processing errors
                    messages = [f"Error processing metadata: {str(e)}"]
            else:
                messages = ["No metadata found for the query."]

            # Save the conversation in the database
            Conversation.objects.create(user=user, user_message=message, model_response=messages)

            # Append the conversation to the history
            conversation_history.append({'user_message': message, 'model_response': messages})

            return JsonResponse({
                "response": messages,
                "conversation_history": conversation_history
            })



def handle_list_experiments(message, user, conversation_history):
            print("In get_metadata function")
            print("In get_metadata function2")
            chatbot_request = ChatbotRequest(query=message)  # Create the request object
            response = get_experiments_list()  # Pass the object to the function

            # Log the response for debugging
            print("Metadata response:")
            print(response)

            # Initialize a list to store messages for multiple entries
            messages = []

            # Check if the response is not empty and process it
            if response:
                try:
                    metadata = response.get('experiments', [])
                    if metadata:
                        # Add the common messages once before looping through experiments
                        messages.append("Connecting to Databricks.\nFetching the model experiment details.\nDownload completed.")
                        
                        # Loop through the experiments if there are multiple entries
                        for experiment in metadata:
                            messages.append(
                                f"Model details: Name - {experiment['name']}, "
                                f"Experiment ID - {experiment['experiment_id']}, "
                                f"Artifact Location - {experiment['artifact_location']}."
                            )
                    else:
                        messages = ["No experiments found in the metadata."]
                except Exception as e:
                    # Handle unexpected data structure or processing errors
                    messages = [f"Error processing metadata: {str(e)}"]
            else:
                messages = ["No metadata found for the query."]

            # Save the conversation in the database
            Conversation.objects.create(user=user, user_message=message, model_response=messages)

            # Append the conversation to the history
            conversation_history.append({'user_message': message, 'model_response': messages})

            return JsonResponse({
                "response": messages,
                "conversation_history": conversation_history
            })
