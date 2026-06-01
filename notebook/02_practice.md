# 02_Practice: BPE vs WordPiece 실전 분석

이 실습에서는 파이썬의 `transformers` 라이브러리를 사용하여 실제 현업에서 쓰이는 **GPT-2(BPE 기반)**와 **BERT(WordPiece 기반)** 토크나이저를 직접 비교해 봅니다.


```python
# 필요 라이브러리 설치 (로컬 환경에 없는 경우 주석 해제 후 실행하세요)
# !pip install transformers
```


```python
from transformers import GPT2Tokenizer, BertTokenizer

# 1. GPT-2 토크나이저 로드 (BPE 방식)
# GPT-2는 기본적으로 BPE 방식을 사용하며, 공백을 Ġ 기호로 표시합니다.
bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 2. BERT 토크나이저 로드 (WordPiece 방식)
# BERT는 WordPiece 방식을 사용하며, 서브워드 앞에 ## 기호를 붙입니다.
wp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]


    C:\Users\정세윤\AppData\Local\Programs\Python\Python313\Lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\정세윤\.cache\huggingface\hub\models--gpt2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]


    C:\Users\정세윤\AppData\Local\Programs\Python\Python313\Lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\정세윤\.cache\huggingface\hub\models--bert-base-uncased. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    


    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


## 비교 1: 신조어/복잡한 영단어 처리


```python
text_eng = "unbelievably"

print("[원본 단어]:", text_eng)
print("-" * 40)
print("[BPE 토큰 분할]:", bpe_tokenizer.tokenize(text_eng))
print("[WordPiece 토큰 분할]:", wp_tokenizer.tokenize(text_eng))
```

    [원본 단어]: unbelievably
    ----------------------------------------
    [BPE 토큰 분할]: ['un', 'bel', 'iev', 'ably']
    [WordPiece 토큰 분할]: ['un', '##bel', '##ie', '##va', '##bly']
    

### 분석 (영단어)
- **BPE**는 빈도 기반이므로 기계적인 바이트 쌍 결합의 흔적이 나타납니다.
- **WordPiece**는 언어학적으로 의미 있는 형태소(예: 접미사 `##ably`) 위주로 쪼개는 성향이 나타나며, 원래 단어를 복원할 때 `##`을 기준으로 합칩니다.

## 비교 2: 한글 처리 (다국어 환경에서의 한계점)


```python
text_kor = "기계 학습을 좋아합니다."

print("[원본 문장]:", text_kor)
print("-" * 40)
print("[BPE 토큰 분할]:", bpe_tokenizer.tokenize(text_kor))
print("[WordPiece 토큰 분할]:", wp_tokenizer.tokenize(text_kor))
```

    [원본 문장]: 기계 학습을 좋아합니다.
    ----------------------------------------
    [BPE 토큰 분할]: ['ê', '¸', '°', 'ê', '³', 'Ħ', 'Ġ', 'íķ', 'Ļ', 'ì', 'Ĭ', 'µ', 'ìĿ', 'Ħ', 'Ġì', '¢', 'ĭ', 'ì', 'ķ', 'Ħ', 'íķ', '©', 'ëĭ', 'Ī', 'ëĭ', '¤', '.']
    [WordPiece 토큰 분할]: ['[UNK]', 'ᄒ', '##ᅡ', '##ᆨ', '##ᄉ', '##ᅳ', '##ᆸ', '##ᄋ', '##ᅳ', '##ᆯ', '[UNK]', '.']
    

### 분석 (한글)
- **BPE (GPT-2)**: 영문 기반으로 학습되었기 때문에 한글을 만나면 글자를 넘어 자음/모음 유니코드 바이트 수준까지 무참히 쪼개버립니다. `Ġ` 기호는 띄어쓰기를 의미합니다.
- **WordPiece (BERT)**: `bert-base-uncased` 역시 영문 전용이므로 한글 단어를 제대로 처리하지 못하고 쪼개집니다. 만약 다국어 모델(mBERT)이나 KoBERT를 사용했다면 형태소 단위로 깔끔하게 잘리며 `##` 기호가 붙습니다.

## 결론
단순히 '텍스트를 자른다'를 넘어, 알고리즘의 **분할 기준(빈도 vs 정보량)**에 따라 전혀 다른 형태의 덩어리가 생성되는 것을 직접 확인했습니다. 기획자는 프로젝트의 언어 타겟과 모델 특성에 맞춰 적절한 토크나이저를 선택해야 합니다.
