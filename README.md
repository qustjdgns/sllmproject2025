# 문제점

google/gemma-2-2b-it
- 빠르고 가볍게 쓸 때는 훌륭하지만, 영어기반 LLM모델의 한계로 한국어 작업에는 어려움을 느낌.
- 감성 표현, 친근한 말투에 약함,Gemma는 포멀하고 기계적인 톤.
- 블로그 글쓰기, 감성 짧은 글쓰기, 상상력 필요한 답변, Gemma는 기계적이고 뻔한 문장을 만들어냄.

--------
# 해결 방법

## google/gemma-2-2b-it 보다 우수한 성능을 가진 한국어 기반 학습 모델 탐색
  
https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard

 Open Ko-LLM Leaderboard는 한국어 대형 언어 모델(LLM)의 성능을 객관적으로 평가하기 위해 Upstage와 한국정보화진흥원(NIA)이 공동으로 구축한 평가 플랫폼

 


## Ko-H5 벤치마크
한국어 LLM의 다양한 능력을 평가하기 위해 Ko-H5라는 벤치마크를 도입했습니다. 이 벤치마크는 다음과 같은 다섯 가지 평가 항목으로 구성되어 있습니다

  Ko-HellaSwag: 일상적 상식과 문맥 이해를 평가합니다.

  Ko-ARC: 초등학생 수준의 과학 문제를 통해 추론 능력을 평가합니다.

  Ko-MMLU: 전문 지식과 다분야 이해도를 측정합니다.

  Ko-TruthfulQA: 사실성 및 허위 정보 생성 여부를 평가합니다.

  Ko-CommonGen v2: 한국어 상식 기반의 문장 생성 능력을 평가합니다.
​
| 모델 | 평균 (Average) | Ko-GPQA | Ko-Winogrande | Ko-GSM8k | Ko-EQ Bench | Ko-IFEval | KorNAT-CKA | KorNAT-SVA | Ko-Harmlessness |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| maywell/Synatra-42dot-1.3B | 31.66 | 24.75 | 55.41 | 2.05 | 0 | 22.65 | 27.31 | 51.01 | 60.44 |
| DILAB-HYU/koquality-polyglot-1.3b | 31.07 | 28.28 | 51.3 | 0.08 | 0 | 25.21 | 24.92 | 51.01 | 59.55 |
| jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.5.3 | 30.93 | 29.8 | 52.09 | 0 | 0 | 14.85 | 31.94 | 51.01 | 64.22 |
| cpm-ai/gemma-ko-v01 | 30.78 | 21.21 | 49.17 | 0.61 | -0.64 | 28.2 | 22.67 | 47.34 | 65.45 |
| Qwen/Qwen2.5-0.5B-Instruct | 30.61 | 21.72 | 51.62 | 0.38 | 1.65 | 25.65 | 14.71 | 50.64 | 62.15 |
| DooDooHyun/AIFT-42dot_LLM-PLM-1.3B-v1.51 | 30.43 | 27.78 | 54.06 | 0 | 0 | 12.51 | 30.41 | 51.01 | 63.31 |
| heegyu/polyglot-ko-1.3b-chat | 30.19 | 25.76 | 51.14 | 0 | 0 | 24.43 | 18.97 | 50.63 | 59.99 |
| Edentns/DataVortexTL-1.1B-v0.1 | 30.04 | 22.73 | 50.91 | 0.99 | 0.47 | 21.32 | 13.1 | 50.83 | 61.73 |
| heegyu/42dot_LLM-PLM-1.3B-mt | 29.99 | 22.22 | 52.8 | 0.08 | 0 | 27.37 | 14.55 | 51.01 | 60.09 |
| Leejy0-0/gemma-2b-it-sum-ko | 29.99 | 20.71 | 51.38 | 0.08 | 0 | 28.59 | 15.18 | 50.61 | 64.75 |

- maywell/Synatra-42dot-1.3B 0~3b 모델 중 가장 점수가 높은 한국어 기반 학습 모델.
- 하단의 gamma2 기반으로 한 파인튜닝보다 높은 점수를 기록.

--------
# maywell/Synatra-42dot-1.3B 모델이란?

![스크린샷 2025-04-26 223929](https://github.com/user-attachments/assets/d5952cbc-663c-4414-99df-27e3f5181474)

    python torch_dtype=torch.float16).to("cuda")
    torch_dtype=torch.float16 

- 모델의 파라미터를 16비트 부동소수점 형식으로 변환하여 메모리 사용을 줄이고 연산 속도를 높입니다. .to("cuda")는 모델을 GPU로 이동시켜 연산 성능을 향상시킵니다.


--------
# maywell/Synatra-42dot-1.3B 입력 조정




## 1.문제점

초기에 아무것도 입력 조정 하지 않을 때 사용자의 질문을 받으면 같은 단어를 반복하고 max_length 길이가 짧다보니 단답형으로 응답합.

## 수정방안

```python
  outputs = model.generate(
            **inputs,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  #  반복 억제 추가
            no_repeat_ngram_size=3,  #  3단어 이상 반복 금지
            temperature=0.7,  #  다양성 확보
            top_p=0.9
```
model.generate() 반복억제, 3단어 이상 반복 금지 기능 활성화

---

## 2.문제점

정확한 정보를 제공 하고 있으나, 응답에 불필요한 요소(공백, 질문)등이 같이 포함됨

## 수정방안

```python
def clean_response(prompt: str, decoded_output: str) -> str:
    response = decoded_output.replace(prompt, "").strip()
    if response.startswith("요."):
        response = response[2:].strip()
    if response.startswith("-"):
        response = response[1:].strip()
    return response
```
응답에 불필요한 요소를 제거하는 함수를 추가하여 깔끔하게 응답이 출력됨.

----

# 출력 예시


![dg](https://github.com/user-attachments/assets/f2ff93f5-4fd1-4a3d-a6e5-6b30db61b3e9)

1592년 4월 13일 일본군에게 포위된 한산도대첩에서 승리하여 전쟁을 승리로 이끌었습니다.-> 1592년 7월 한산도대첩에서 일본 수군을 크게 무찌르며 전세를 뒤집었습니다.
를 제외하면 대체로 정확한 정보를 약 40초 이내에 응답하고 있음.


# 결론
- gamma2 보단 한국어에 강점을 보이나 추론능력이 살짝 아쉽게 느껴짐
- 하지만 빠른속도를 감안하면 성능에 대체로 만족
- 추후 추론 능력과 지식 학습을 강화하면 경량화 모델 중에선 견줄 상대가 없어보임

# main.py

```python
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
'''





