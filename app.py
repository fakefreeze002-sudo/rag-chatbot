import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / ".chroma"
COLLECTION_NAME = "hwaseong_civil_documents"
load_dotenv(BASE_DIR / ".env")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
LOCAL_DISTANCE_THRESHOLD = 0.55
AMBIGUITY_MARGIN = 0.07


def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일에 설정하세요.")
    return OpenAI(api_key=key)


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split at paragraph boundaries when possible, with a small overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size - overlap :]
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def topic_for(filename: str, text: str) -> str:
    value = f"{filename} {text[:250]}"
    if "주차" in value:
        return "주차"
    if any(word in value for word in ("동물", "반려", "길고양이", "유기")):
        return "동물보호"
    if any(word in value for word in ("폐기물", "쓰레기", "종량제", "재활용")):
        return "폐기물 관리"
    if "민원" in value:
        return "민원 처리"
    return "기타"


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    return [item.embedding for item in client.embeddings.create(model=EMBEDDING_MODEL, input=texts).data]


@st.cache_resource(show_spinner=False)
def build_or_load_collection() -> Any:
    client = get_openai_client()
    chroma = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    records = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(split_text(text)):
            digest = hashlib.sha256(f"{path.name}:{index}:{chunk}".encode()).hexdigest()
            records.append((digest, chunk, {"source": path.name, "topic": topic_for(path.name, chunk), "chunk": index}))

    existing = set(collection.get(include=[])["ids"])
    missing = [record for record in records if record[0] not in existing]
    for start in range(0, len(missing), 64):
        batch = missing[start : start + 64]
        vectors = embed(client, [item[1] for item in batch])
        collection.upsert(
            ids=[item[0] for item in batch],
            documents=[item[1] for item in batch],
            metadatas=[item[2] for item in batch],
            embeddings=vectors,
        )
    return collection


def normalize_question(client: OpenAI, question: str) -> tuple[str, bool]:
    """Only correct clear typos/grammar; don't alter the user's intent."""
    prompt = """다음 한국어 민원 질문의 명백한 오타와 비문만 고치세요. 의미, 조건, 대상은 절대 추가하거나 바꾸지 마세요.
JSON만 출력: {\"normalized\": \"...\", \"needs_confirmation\": true|false}
질문: """ + question
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    normalized = str(data.get("normalized", question)).strip() or question
    return normalized, bool(data.get("needs_confirmation", normalized != question))


def search_local(collection: Any, client: OpenAI, question: str) -> list[dict[str, Any]]:
    result = collection.query(
        query_embeddings=embed(client, [question]),
        n_results=6,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "meta": meta, "distance": distance}
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def ambiguous_topics(results: list[dict[str, Any]]) -> list[str]:
    relevant = [r for r in results if r["distance"] <= LOCAL_DISTANCE_THRESHOLD]
    if len(relevant) < 2:
        return []
    best_by_topic: dict[str, float] = {}
    for result in relevant:
        topic = result["meta"]["topic"]
        best_by_topic[topic] = min(best_by_topic.get(topic, 1.0), result["distance"])
    ordered = sorted(best_by_topic.items(), key=lambda item: item[1])
    if len(ordered) > 1 and ordered[1][1] - ordered[0][1] <= AMBIGUITY_MARGIN:
        return [topic for topic, _ in ordered[:3]]
    return []


def local_answer(client: OpenAI, question: str, results: list[dict[str, Any]]) -> str:
    evidence = "\n\n---\n\n".join(r["text"] for r in results[:4])
    prompt = f"""당신은 화성시 민원 안내 도우미입니다. 아래 제공 문서 발췌만 근거로 한국어로 답변하세요.
문서에 직접 뒷받침되지 않는 내용은 추정하지 마세요. 관련 내용을 찾지 못하면 정확히 '자료에서 확인할 수 없습니다'라고 답하세요.

질문: {question}

문서 발췌:
{evidence}"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "자료에서 확인할 수 없습니다"


def web_search(question: str) -> list[dict[str, str]]:
    try:
        return list(DDGS().text(f"화성시 {question}", region="kr-kr", max_results=5))
    except Exception:
        return []


def web_answer(client: OpenAI, question: str, results: list[dict[str, str]]) -> str:
    excerpts = "\n\n".join(f"제목: {r.get('title', '')}\n내용: {r.get('body', '')}" for r in results)
    prompt = f"""아래 웹 검색 발췌만 근거로 질문에 짧게 답하세요. 발췌에 없는 사실은 추가하지 마세요.
질문: {question}\n\n웹 발췌:\n{excerpts}"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "자료에서 확인할 수 없습니다"


def show_sources(results: list[dict[str, Any]]) -> None:
    sources = list(dict.fromkeys(result["meta"]["source"] for result in results))
    st.caption("출처 파일: " + ", ".join(sources))
    with st.expander("검색된 문서 발췌 보기"):
        for result in results:
            st.markdown(f"**{result['meta']['source']}** · {result['meta']['topic']}")
            st.write(result["text"])


st.set_page_config(page_title="화성시 민원 챗봇", page_icon="🏙️")
st.title("🏙️ RAG 기반 화성시 민원 챗봇")
st.caption("data 폴더의 문서만으로 답변하며, 자료에 없을 때만 외부 웹 검색 결과를 별도로 표시합니다.")

if "question_to_search" not in st.session_state:
    st.session_state.question_to_search = ""

question = st.text_input("민원 질문을 입력하세요", value=st.session_state.question_to_search, placeholder="예: 대형폐기물은 어떻게 버리나요?")
search_clicked = st.button("질문하기", type="primary")

try:
    openai_client = get_openai_client()
    collection = build_or_load_collection()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if search_clicked and question.strip():
    normalized, needs_confirmation = normalize_question(openai_client, question.strip())
    if needs_confirmation:
        st.session_state.pending_normalized = normalized
        st.info(f"다음과 같이 정리한 질문이 맞나요?\n\n**{normalized}**")
    else:
        st.session_state.question_to_search = normalized
        st.session_state.run_search = True

if "pending_normalized" in st.session_state:
    col1, col2 = st.columns(2)
    if col1.button("네, 이 질문으로 검색", key="confirm_normalized"):
        st.session_state.question_to_search = st.session_state.pop("pending_normalized")
        st.session_state.run_search = True
        st.rerun()
    if col2.button("아니요, 다시 입력", key="reject_normalized"):
        del st.session_state.pending_normalized
        st.session_state.run_search = False

if st.session_state.pop("run_search", False):
    active_question = st.session_state.question_to_search
    with st.spinner("문서를 검색하고 있습니다..."):
        local_results = search_local(collection, openai_client, active_question)
    relevant = [r for r in local_results if r["distance"] <= LOCAL_DISTANCE_THRESHOLD]
    choices = ambiguous_topics(local_results)
    if choices and "clarified" not in active_question:
        st.warning("서로 다른 유형의 안내가 함께 검색되었습니다. 어느 분야인지 선택해 주세요.")
        selected = st.radio("추가 확인", choices, horizontal=True)
        if st.button("선택한 분야로 다시 검색"):
            st.session_state.question_to_search = f"{active_question} (분야: {selected}, clarified)"
            st.session_state.run_search = True
            st.rerun()
    elif relevant:
        st.subheader("답변")
        st.write(local_answer(openai_client, active_question, relevant))
        show_sources(relevant)
    else:
        st.info("로컬 문서에 관련 내용을 찾지 못해 외부 웹을 검색했습니다.")
        with st.spinner("외부 검색 결과를 확인하고 있습니다..."):
            external = web_search(active_question)
        if external:
            st.subheader("외부 출처 답변")
            st.write(web_answer(openai_client, active_question, external))
            st.caption("⚠️ 아래 링크는 로컬 문서가 아닌 외부 웹 출처입니다.")
            for result in external:
                st.markdown(f"- [{result.get('title', '검색 결과')}]({result.get('href', '')})")
        else:
            st.subheader("답변")
            st.write("자료에서 확인할 수 없습니다")
