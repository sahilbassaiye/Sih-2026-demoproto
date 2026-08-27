First download all the requirements file
     pip install -r requirements.txt
Secondly install en_core_web
     python -m spacy download en_core_web_sm
For whisper to work install FFmpeg for audio only
run uvicorn main:app --reload --port 8000 in bakend folder
