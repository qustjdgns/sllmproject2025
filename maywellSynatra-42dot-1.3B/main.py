import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn



os.environ['HF_TOKEN'] = ""

# FastAPI 객체 생성 (중복된 선언 제거)
app = FastAPI()

# CORS 설정: 모든 출처 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # '*'은 모든 출처를 허용합니다. 프로덕션에서는 특정 출처로 제한하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("maywell/Synatra-42dot-1.3B")
model = AutoModelForCausalLM.from_pretrained(
    "maywell/Synatra-42dot-1.3B",
    torch_dtype=torch.float16
).to("cuda")


# 응답 클리닝 함수
def clean_response(prompt: str, decoded_output: str) -> str:
    response = decoded_output.replace(prompt, "").strip()
    if response.startswith("요."):
        response = response[2:].strip()
    if response.startswith("-"):
        response = response[1:].strip()
    return response

# 모델로 응답 생성
def generate_gemma_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  #  반복 억제 추가
            no_repeat_ngram_size=3,  #  3단어 이상 반복 금지
            temperature=0.2,  #  다양성 확보
            top_p=0.9
     )



    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = clean_response(prompt, decoded)
    return final_response

# 요청 모델 정의
class PromptRequest(BaseModel):
    prompt: str

# 기본 GET 엔드포인트
@app.get("/")
async def root():
    return {"message": "Hello Chatting"}

# POST 엔드포인트 (응답 생성)
@app.post("/answer")
async def answer(prompt_request: PromptRequest):
    print(f"Received request: {prompt_request}")
    prompt = prompt_request.prompt
    result = generate_gemma_response(prompt)
    return {"result": result}

# 서버 실행
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
