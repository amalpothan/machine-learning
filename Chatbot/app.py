import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from main import get_response

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/chatbot/user-message")
def PredictIntent(input_test: InputText):
    output = get_response(input_test.text)
    return {"result":output}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=5000)