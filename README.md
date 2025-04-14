# exllamav3-openai-server
An very basic OpenAI Compatible API server that is compatible with exllamav3 quantization formats. Works with most front ends like OpenWebUI or SillyTavern. 

# Requirements

(Optional) Create a python venv:  
`pytyhon3 -m venv .venv`  
`source .venv/bin/activate`  

Install exllamav3 requirements:  
`pip install -r https://raw.githubusercontent.com/turboderp-org/exllamav3/master/requirements.txt`  
Then install exllamav3 itself:  
`pip install git+https://github.com/turboderp-org/exllamav3.git`  
Then you can install the requirements for this repo:  
`pip install fastapi uvicorn pydantic`  

# Usage
`python server.py -m /path/to/model --host 0.0.0.0 --port 5000 --mode llama3`

# Acknowledgements
Turboderp [exllamav3](https://github.com/turboderp-org/exllamav3)
