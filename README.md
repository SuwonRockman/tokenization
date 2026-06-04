# 🧩 토크나이제이션: 텍스트를 LLM이 이해하는 언어로 변환하기

[![Generated](https://img.shields.io/badge/Date-2026.05.31-blue.svg)](#) [![Difficulty](https://img.shields.io/badge/Difficulty-Medium%20High-orange.svg)](#)

> [!NOTE]
> **핵심 질문:** LLM은 텍스트가 아닌 숫자만 이해합니다. 그렇다면 "machine learning"이라는 단어를 어떻게 숫자로 변환할까요? 그리고 "machines", "learning", "learnings" 같은 변형된 단어들은 어떻게 처리할까요?

---

## 📌 배경: 왜 토크나이제이션이 어려운가?

### 사람 vs LLM의 관점 차이

LLM이 텍스트를 처리하려면 **텍스트를 작은 단위로 나누고 숫자로 변환**해야 합니다. 단어를 어떻게 나눌지에 대한 3가지 선택지를 살펴봅니다.

<table style="width: 100%; display: table;">
  <thead>
    <tr>
      <th align="center" width="25%">관점 / 접근 방식</th>
      <th align="center" width="35%"><code>machine</code> 처리 예시</th>
      <th align="center" width="40%">설명 및 한계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>사람의 관점</b></td>
      <td align="center">의미 단위로 직관적 이해</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><b>선택지 1: 문자 단위</b></td>
      <td align="center"><code>['m','a','c','h','i','n','e']</code><br>→ <code>[1,2,3,4,5,6,7]</code></td>
      <td align="center"><b>7개 토큰 생성.</b> 정보가 너무 분산되어 비효율적</td>
    </tr>
    <tr>
      <td align="center"><b>선택지 2: 단어 단위</b></td>
      <td align="center"><code>['machine']</code></td>
      <td align="center">단어 변형(<code>machines</code>)마다 새로운 ID 필요.<br><b>어휘 폭발(Vocabulary 낭비)</b> 발생</td>
    </tr>
    <tr>
      <td align="center"><b>선택지 3: 서브워드</b></td>
      <td align="center">❓</td>
      <td align="center"><b>목표:</b> 적절한 크기의 단위 찾기</td>
    </tr>
  </tbody>
</table>

### ⚠️ 실제 언어별 영향

단발성 예시 문장을 기준으로 비교해 보면, 언어별로 **글자당 정보 압축률(효율성)**에서 압도적인 차이가 발생합니다.

<table style="width: 100%; display: table;">
  <thead>
    <tr>
      <th align="center" width="12%">언어</th>
      <th align="center" width="18%">예시 문장</th>
      <th align="center" width="15%">실제 글자 수</th>
      <th align="center" width="25%">서브워드 토큰 수</th>
      <th align="center" width="15%">글자당 토큰 소모율<br>(압축비)</th>
      <th align="center" width="15%">누적 사용 시 실제 API 비용 차이</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>영어</b></td>
      <td align="center"><code>hello world</code></td>
      <td align="center">11글자</td>
      <td align="center"><b>2개</b><br>(<code>['hello', 'world']</code>)</td>
      <td align="center"><b>0.18개</b><br>(가장 효율적)</td>
      <td align="center"><b>1배</b> (기준)</td>
    </tr>
    <tr>
      <td align="center"><b>한글</b></td>
      <td align="center"><code>안녕하세요</code></td>
      <td align="center">5글자</td>
      <td align="center"><b>2개</b><br>(<code>['안녕', '하세요']</code>)</td>
      <td align="center"><b>0.40개</b><br>(보통)</td>
      <td align="center"><b>약 2.5배 비쌈</b></td>
    </tr>
    <tr>
      <td align="center"><b>중국어</b></td>
      <td align="center"><code>你好</code></td>
      <td align="center">2글자</td>
      <td align="center"><b>2개</b><br>(<code>['你', '好']</code>)</td>
      <td align="center"><b>1.00개</b><br>(비효율적)</td>
      <td align="center"><b>약 3.0배 비쌈</b></td>
    </tr>
  </tbody>
</table>

> [!NOTE]
> **📊 글자당 토큰 소모율 계산 로직**
> * **글자당 토큰 소모율 계산식:** `서브워드 토큰 수 ÷ 실제 글자 수`
>   - **영어:** 2개 토큰 ÷ 11글자 = **0.18개**
>   - **한글:** 2개 토큰 ÷ 5글자 = **0.40개**
>   - **중국어:** 2개 토큰 ÷ 2글자 = **1.00개**

<details>
<summary><b>🤔 질문: 셋 다 똑같이 '2토큰'인데 왜 API 요금이 2.5배~3배나 차이 난다고 하나요? (표 해설)</b></summary>

### 💡 핵심: "정보의 압축 효율"과 "누적 사용량"의 차이 때문입니다!

#### 1. 단발성 2토큰 요금 자체는 똑같습니다.
위의 짧은 예시(`hello world`, `안녕하세요`, `你好`)를 모델에 딱 한 번만 전송할 때는 셋 다 똑같이 **2토큰 요금**만 나옵니다. 단발성 요금은 동일한 것이 맞습니다.

#### 2. 하지만 "전달한 글자 수 대비 소모율(압축비)"이 완전히 다릅니다.
* **영어:** 11글자(단어 2개)라는 넉넉한 양의 정보를 단 2토큰으로 압축해냈습니다. (글자당 0.18토큰 소모)
* **한글:** 단어로는 1개, 글자 수로는 영어의 절반도 안 되는 5글자인데 벌써 영어와 똑같은 2토큰을 다 써버렸습니다. (글자당 0.40토큰 소모)
* **중국어:** 단 2글자인데 벌써 2토큰을 다 써버렸습니다. (글자당 1.00토큰 소모)

#### 3. 이 차이가 실제 긴 대화(누적 사용)로 이어지면 요금 폭탄이 됩니다.
사용자가 LLM을 쓸 때는 짧은 단어가 아니라 보통 수백~수천 글자의 긴 대화를 주고받습니다. 예를 들어 **똑같이 1,000글자 분량의 정보**를 모델에 보낸다면:
* **영어:** 약 **180 토큰** 소모 (글자당 소모율 0.18개 기준)
* **한글:** 약 **450 토큰** 소모 (영어보다 더 많이 쪼개지며 평균적으로 **영어의 2.5배** 요금 발생)
* **중국어:** 약 **550 토큰** 소모 (한자마다 개별 토큰으로 깨져서 평균적으로 **영어의 3배** 요금 발생)

**결론:** 단발성 예시의 토큰 수(2개)는 동일하지만, **글자당 정보 압축률의 극심한 효율 차이** 때문에 실제 장문이나 누적 API 사용 환경에서는 영어 대비 한글은 2.5배, 중국어는 3배 더 많은 비용을 내게 되는 것입니다.

</details>

> [!WARNING]
> **결론:** 토크나이제이션 방식의 선택에 따라 **처리 비용이 최대 3배 이상 차이**납니다!

---

## 🔍 핵심 개념: 적절한 크기의 단위 찾기

토크나이제이션의 궁극적인 목표는 **"텍스트의 의미를 최대한 보존하면서, 가장 효율적인 크기의 단위로 나누는 것"**입니다.

<details>
<summary><b>단위별 장단점 비교 (클릭해서 펼치기)</b></summary>

*   **문자 단위 (너무 작음):** 정보가 분산되고 시퀀스가 길어져 비효율적입니다.
*   **단어 단위 (너무 큼):** 어휘 사전이 무한정 커지고 OOV(Out-Of-Vocabulary) 문제가 다수 발생합니다.
*   **서브워드 토크나이제이션 (최적):** `['hel', 'lo']` 처럼 쪼개어 의미와 효율의 균형을 잡습니다.

</details>

---

## 💡 3가지 솔루션과 기술적 차이

### 1️⃣ BPE (Byte Pair Encoding) - 빈도 기반

텍스트 데이터에서 가장 자주 등장하는 문자나 바이트 쌍을 반복적으로 병합하여 하나의 토큰으로 만들어가는 **상향식(Bottom-Up) 서브워드 분할 알고리즘**입니다. **(GPT-2, GPT-3, GPT-4, Llama)**

*   **동작 원리:**
    1. 모든 단어를 문자(Character) 단위로 분할하여 기본 사전을 구성합니다.
    2. 말뭉치에서 가장 빈번하게 인접하여 나타나는 두 토큰 쌍을 찾아 하나의 토큰으로 합칩니다.
    3. 지정한 어휘 사전 크기(Vocabulary Size)에 도달할 때까지 이 병합 과정을 탐욕적(Greedy)으로 반복합니다.
*   **💡 동작 예시 (병합 과정):**
    - 입력 단어: `'l', 'o', 'w', 'e', 'r'`
    - **1단계 (기본 문자 분할):** `['l', 'o', 'w', 'e', 'r']`
    - **2단계 ('e'와 'r'이 빈도 1위 ➔ 'er' 병합):** `['l', 'o', 'w', 'er']`
    - **3단계 ('l'과 'o'가 빈도 2위 ➔ 'lo' 병합):** `['lo', 'w', 'er']`
    - **4단계 ('lo'와 'w'가 빈도 3위 ➔ 'low' 병합):** `['low', 'er']`
*   **기술적 심화 특징:**
    - **탐욕적 병합(Greedy Merge):** 한 번 합친 병합 규칙은 고정되어 번복되지 않습니다. 빈도만을 기준으로 삼기 때문에 실제 단어의 언어학적 의미는 완전히 무시됩니다.
    - **OOV(Out-of-Vocabulary) 방지:** 학습 시 보지 못한 신조어가 등장하더라도, 가장 작은 기초 문자 단위까지 쪼개서 표현하므로 에러가 나지 않고 어떻게든 인식해 냅니다.
    - **공백 인코딩:** 단어의 경계를 보존하기 위해 공백을 특수 문자 `Ġ` (Llama 등에서는 `_`)로 변환하여 처리합니다.
*   **어휘 크기:** 약 50,000개 (50K)

> [!CAUTION]
> **한글 처리 문제 (BPE의 한계)**
> "기계 학습을 좋아합니다" → `['기', '계', '학', '습', '을', '좋', '아', '합', '니', '다']` (10개 토큰)
> **한계 분석:** BPE는 빈도 기준이므로 다국어 코퍼스에서 한글 단어 빈도가 낮으면, 한글 단어를 통째로 등록하지 못하고 자음/모음 수준까지 산산조각 냅니다. 결과적으로 토큰 수가 폭증하고 의미 보존율이 극도로 저하됩니다.

---

### 2️⃣ WordPiece - 확률 기반

단순히 빈도가 높은 쌍을 합치지 않고, 두 서브워드를 합쳤을 때 **전체 코퍼스의 우도(Likelihood, 발생 확률)를 가장 크게 높이는 조합**을 선택하는 알고리즘입니다. **(BERT, RoBERTa, KoBERT)**

*   **동작 원리:**
    1. BPE처럼 기초 문자 단위에서 시작하여 병합 후보군을 만듭니다.
    2. 후보군 중 두 토큰 $A$와 $B$를 합쳤을 때, 개별 토큰의 확률곱($P(A) \times P(B)$) 대비 합쳐진 토큰의 확률($P(AB)$)이 가장 극대화되는(즉, 정보 획득량이 가장 높은) 쌍을 병합합니다.
    3. 이를 통해 말뭉치 전체의 언어 모델 확률을 최대화하는 최적의 서브워드 사전을 도출합니다.
*   **💡 동작 예시 (## 결합 분할):**
    - 입력 단어: `'playing'` (사전에 없는 단어인 경우)
    - 접두사 `##`을 이용해 단어 뒷부분부터 가장 널리 쓰이는 형태소 형태의 결합을 분리해 냅니다.
    - **최종 토큰 결과:** `['play', '##ing']` (어근과 어미 형태소 결합을 완벽하게 인지함)
*   **기술적 심화 특징:**
    - **서브워드 접두사 `##`:** 단어의 중간이나 끝에 결합하는 서브워드에는 접두사 `##`를 붙여 단어 내 위치를 명시합니다 (예: `learning` ➔ `['learn', '##ing']`). 이는 토큰 분할 후 원본 단어로 복원할 때 단어 경계를 정확히 구분해 줍니다.
    - **사전 기반 견고성:** BPE보다 사전 구성이 다소 보수적이지만, 문맥적 연관성이 매우 높고 효율적인 의미 단위를 추출하는 데 뛰어납니다.
*   **어휘 크기:** 약 30,000개 (30K)

---

### 3️⃣ SentencePiece - 언어 독립적

텍스트를 미리 공백(띄어쓰기) 기준으로 나누는 **사전 토크나이저(Pre-tokenizer)를 사용하지 않고**, 입력 텍스트 전체를 하나의 긴 바이트 스트림으로 취급하는 언어 독립적 서브워드 분할 툴킷입니다. **(mBART, XLM-RoBERTa, Llama 2)**

*   **동작 원리:**
    1. BPE 또는 Unigram 언어 모델 알고리즘을 코어로 사용합니다.
    2. 띄어쓰기를 포함한 원시 텍스트(Raw Text)를 그대로 입력받아, 공백조차 하나의 일반 문자인 `▁` (메타스페이스) 기호로 취급하여 인코딩합니다.
    3. 입력 텍스트의 언어가 무엇이든(한글, 영어, 중국어 등) 사전 분할 규칙 없이 동일한 기준으로 바이트 레벨에서 서브워드를 찾아냅니다.
*   **💡 동작 예시 (메타스페이스 치환 분할):**
    - 입력 문장: `"get up"` (사전 분할 없이 띄어쓰기째로 입력)
    - **1단계 (공백을 메타스페이스로 교체):** `"get▁up"`
    - **2단계 (코어 알고리즘에 따른 서브워드 추출):** `['get', '▁up']` (공백 자체를 뒤 단어에 밀착시켜 하나의 문자적 속성으로 다룸)
*   **기술적 심화 특징:**
    - **사전 토크나이저 불필요(No Pre-tokenization):** 띄어쓰기가 없는 중국어/일본어나 조사 결합이 복잡한 한국어처럼 띄어쓰기 기준 분할이 어려운 언어들을 학습할 때 규칙 예외 없이 완벽하게 처리할 수 있습니다.
    - **바이트 폴백(Byte Fallback):** 사전에 없는 희귀한 유니코드 문자나 이모지가 입력되면, UTF-8 바이트 값(예: `[0xEa]`)으로 하위 변환하여 토큰을 생성하므로 OOV 에러가 원천 차단됩니다.
*   **어휘 크기:** 약 32,000개 (32K)

> [!TIP]
> **한글 처리에 최적화된 이유!**
> "기계 학습을 좋아합니다" → `['▁기계', '▁학습', '을', '▁좋아합니다']` (5개 토큰)
> **장점 분석:** 띄어쓰기 공백을 메타스페이스(`▁`)로 일반 문자 취급하므로 한국어의 조사 탈착 과정에서 문법 구조가 깨지지 않고, 자음/모음 분리 없이 단어의 의미적 핵심 형태를 그대로 토큰화하여 정보 보존 효율이 대단히 높습니다.

---

## 📊 실제 영향 및 성능 분석

### 1. 처리 비용 (토큰 수 차이)

같은 의미의 한글 문장("기계 학습을 좋아합니다")을 처리할 때 모델별 API 비용은 다음과 같습니다.

<table style="width: 100%; display: table;">
  <thead>
    <tr>
      <th align="center" width="25%">토크나이저</th>
      <th align="center" width="25%">처리 토큰 수</th>
      <th align="center" width="25%">예상 API 비용</th>
      <th align="center" width="25%">비고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>BPE</b> (GPT-2)</td>
      <td align="center">10개</td>
      <td align="center"><code>$0.005</code></td>
      <td align="center">🚨 <b>가장 비쌈</b></td>
    </tr>
    <tr>
      <td align="center"><b>WordPiece</b> (KoBERT)</td>
      <td align="center">6개</td>
      <td align="center"><code>$0.003</code></td>
      <td align="center">보통</td>
    </tr>
    <tr>
      <td align="center"><b>SentencePiece</b> (XLM)</td>
      <td align="center"><b>5개</b></td>
      <td align="center"><b><code>$0.0025</code></b></td>
      <td align="center">✅ <b>BPE 대비 절반 가격</b></td>
    </tr>
  </tbody>
</table>

### 2. 다국어 서비스 공정성

영문 기반 모델(BPE)에서는 "Hello"가 **1 토큰**인 반면, "안녕하세요"는 **5 토큰**으로 처리되어 언어별 비용 편차가 5배에 달합니다. 반면 **SentencePiece**는 "안녕하세요"를 3 토큰으로 처리하여 언어 간 편차를 최소화합니다.

### 3. 모델 성능(Attention)에 미치는 영향

> [!IMPORTANT]
> **Attention 계산량 = $O(n^2)$**
> 토큰 수가 2배 증가하면 모델의 계산량은 **4배 증가**합니다. 또한 메모리 점유율이 2배 높아져 배치 사이즈(Batch Size)를 4분의 1로 줄여야 하는 치명적 병목이 발생할 수 있습니다.

---

## 🗂️ 프로젝트 구조

```text
tokenization/
├── README.md                           # 📖 기술 깊이 중심 가이드 (현재 문서)
├── .gitignore
├── notebook/
│   ├── 01_theory.md                   # 📘 이론 (BPE vs WordPiece 심층 분석)
│   └── 03_advanced_practice.ipynb              # 💻 실습 주피터 노트북 (BPE, WordPiece 비용 분석)
└── scripts/
    └── (예정)
```

---

## 💼 실무 교훈: 어떤 토크나이저를 선택해야 할까?

<table style="width: 100%; display: table;">
  <thead>
    <tr>
      <th align="center" width="30%">프로젝트 상황</th>
      <th align="center" width="30%">추천 토크나이저</th>
      <th align="center" width="40%">핵심 이유</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>영어로만 구성된 서비스</b></td>
      <td align="center"><code>BPE</code> (GPT 모델)</td>
      <td align="center">가장 널리 쓰이며 영문 텍스트 압축에 매우 효율적</td>
    </tr>
    <tr>
      <td align="center"><b>한글이 주력인 서비스</b></td>
      <td align="center"><code>WordPiece</code> (KoBERT) / <code>SentencePiece</code></td>
      <td align="center">자음/모음 분리 방지. <b>BPE 절대 지양!</b></td>
    </tr>
    <tr>
      <td align="center"><b>글로벌 / 다국어 서비스</b></td>
      <td align="center"><code>SentencePiece</code> 필수</td>
      <td align="center">특정 언어에 종속되지 않아 언어 간 비용 공정성 확보</td>
    </tr>
  </tbody>
</table>

---

## 🚀 다음 단계 (Next Step)

지금까지 기획자의 관점에서 토크나이제이션이 비용과 성능에 미치는 전반적인 영향(개괄)을 살펴보았습니다. 
**[2단계 Theory (심층 이론)](./notebook/01_theory.md)** 파트에서는 이 중 현대 NLP의 근간이 된 두 모델(**BPE**와 **WordPiece**)의 알고리즘 철학과 수학적 차이를 본격적으로 해부합니다.

<div align="right">
  <b>GitHub:</b> <a href="https://github.com/SuwonRockman/tokenization">SuwonRockman/tokenization</a><br>
</div>
