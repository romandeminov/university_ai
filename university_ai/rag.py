from pathlib import Path
import json
import numpy as np
from ollama import embed, chat

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "university-qwen14b"

TOP_K = 5

HISTORY_FOR_SEARCH = 4

def load_knowledge_base():
    file_path = DATA_DIR / "knowledge.json"

    if not file_path.exists():
        raise FileNotFoundError(
            "База знаний не найдена. "
            "Сначала запустите ingest.py"
        )

    return json.loads(
        file_path.read_text(encoding="utf-8")
    )


def load_system_prompt():
    assistant_file = KNOWLEDGE_DIR / "assistant.md"

    if not assistant_file.exists():
        raise FileNotFoundError(
            "Файл assistant.md не найден."
        )

    return assistant_file.read_text(
        encoding="utf-8"
    )


DOCUMENTS = load_knowledge_base()
SYSTEM_PROMPT = load_system_prompt()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator

def search_knowledge(
    query,
    documents,
    top_k=TOP_K
):
    response = embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = []

    for document in documents:
        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "text": document["text"],
            "source": document["source"],
            "score": similarity
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]

def build_context(results):
    parts = []

    for result in results:
        parts.append(
            f"[{result['source']}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(parts)

def build_search_query(question, history):
    if not history:
        return question.strip()

    previous_questions = []

    for message in history[-HISTORY_FOR_SEARCH:]:
        if message.get("role") != "user":
            continue

        content = message.get(
            "content",
            ""
        ).strip()

        if content:
            previous_questions.append(content)

    if not previous_questions:
        return question.strip()

    previous_text = "\n".join(
        previous_questions
    )

    return (
        f"{previous_text}\n"
        f"{question}"
    )

def build_history_text(history):
    if not history:
        return ""

    parts = []

    for message in history[-10:]:
        role = message.get("role")
        content = message.get(
            "content",
            ""
        ).strip()

        if not content:
            continue

        if role == "user":
            parts.append(
                f"Пользователь: {content}"
            )

        elif role == "assistant":
            parts.append(
                f"Ассистент: {content}"
            )

    return "\n".join(parts)

def ask(
    question,
    history,
    user_role=None
):

    search_query = build_search_query(
        question,
        history
    )

    results = search_knowledge(
        search_query,
        DOCUMENTS
    )

    context = build_context(results)

    history_text = build_history_text(history)

    role_text = user_role or "неизвестна"

    prompt = f"""
Ответь на вопрос пользователя.

Роль: {role_text}

Используй только информацию из базы знаний.
Не выдумывай данные.
Если ответа нет в базе знаний, напиши:
«В базе знаний нет информации по этому вопросу».

Учитывай историю, если текущий вопрос является продолжением разговора.
Отвечай кратко и понятно на русском языке.
Не добавляй служебные пояснения.

История:
{history_text}

База знаний:
{context}

Вопрос:
{question}
""".strip()

    stream = chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        stream=True,
        keep_alive="10m",
        options={
            "temperature": 0.2,
            "num_predict": 200,
            "num_ctx": 2048
        }
    )

    return stream, results