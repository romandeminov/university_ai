from pathlib import Path
import json
import re

from ollama import embed


BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
ALLOWED_FILES = [
    "features.md",
    "user-guides.md",
    "roles-and-access.md",
    "system.md",
]


def split_text(text: str):
    sections = re.split(
        r"(?=^#{1,6}\s)",
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
            continue

        paragraphs = section.split("\n\n")
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(current) + len(paragraph) + 2 <= CHUNK_SIZE:
                current += (
                    "\n\n" if current else ""
                ) + paragraph
            else:
                if current:
                    chunks.append(current)

                overlap = (
                    current[-CHUNK_OVERLAP:]
                    if current
                    else ""
                )

                current = (
                    overlap
                    + "\n\n"
                    + paragraph
                )

        if current:
            chunks.append(current)

    return chunks


def load_documents():
    documents = []

    for filename in sorted(ALLOWED_FILES):
        file_path = KNOWLEDGE_DIR / filename

        if not file_path.is_file():
            print(
                f"Предупреждение: файл {filename} не найден."
            )
            continue

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = split_text(text)

        for chunk in chunks:
            documents.append({
                "text": chunk,
                "source": filename
            })

    return documents


def create_embeddings(documents):
    texts = [
        document["text"]
        for document in documents
    ]

    print(
        f"Создание embeddings для "
        f"{len(texts)} фрагментов..."
    )

    response = embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = response["embeddings"]

    for document, embedding in zip(
        documents,
        embeddings
    ):
        document["embedding"] = embedding

    return documents


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Загрузка базы знаний...")

    documents = load_documents()

    print(
        f"Найдено фрагментов: "
        f"{len(documents)}"
    )

    if not documents:
        raise RuntimeError(
            "Не найдено ни одного справочного "
            "Markdown-файла в knowledge_base."
        )

    documents = create_embeddings(documents)

    output_file = DATA_DIR / "knowledge.json"

    output_file.write_text(
        json.dumps(
            documents,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("Готово!")
    print(f"База сохранена: {output_file}")


if __name__ == "__main__":
    main()
