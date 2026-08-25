# RAG 기반 화성시 민원 챗봇

`data/`의 TXT 문서를 OpenAI 임베딩으로 벡터화해 ChromaDB에 저장하고, 질문과 가까운 문서 발췌만을 근거로 답변하는 Streamlit 앱입니다.

## 기능

- OpenAI `text-embedding-3-small` 임베딩과 영구 저장형 ChromaDB 사용
- 검색된 로컬 문서에 근거한 답변 및 출처 파일명·발췌 표시
- 로컬 문서의 관련성이 낮으면 DDGS(DuckDuckGo) 웹 검색으로 폴백하고, **외부 출처 답변**과 링크를 분리 표시
- 비문·오타가 감지되면 정리된 질문을 먼저 보여 주고 검색 전 확인
- 비슷한 점수로 서로 다른 분야(주차·동물보호·폐기물 관리)가 검색되면 분야를 선택하는 추가 질문
- 관련 문서와 외부 검색 결과가 모두 없으면 `자료에서 확인할 수 없습니다` 표시

## 설치

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .rag-venv
.\.rag-venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` 파일의 `OPENAI_API_KEY`에 본인의 OpenAI API 키를 넣습니다. API 키는 코드나 Git에 저장하지 않습니다.

```env
OPENAI_API_KEY=sk-...
# 선택: 기본값을 바꾸고 싶을 때만 설정
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

## 실행

```powershell
streamlit run app.py
```

처음 실행하면 `data/*.txt`가 청크 단위로 임베딩되어 `.chroma/`에 저장됩니다. 이후에는 새 문서 또는 변경된 청크만 추가합니다. 인덱스를 처음부터 만들려면 앱을 종료한 뒤 `.chroma` 폴더를 삭제하고 다시 실행하세요.

## 유의사항

- 로컬 답변은 검색된 TXT 발췌만으로 생성하도록 프롬프트가 제한되어 있습니다.
- 웹 검색은 로컬 문서에 관련 내용이 없을 때만 실행되며, 답변과 링크에 외부 출처임을 표시합니다. 검색 결과는 최신 정보 검증이 필요할 수 있습니다.
- 교육용 데이터에는 예시 문서가 포함될 수 있으므로 실제 민원 처리 전에는 공식 화성시 안내를 확인하세요.
