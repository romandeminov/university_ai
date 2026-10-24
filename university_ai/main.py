from rag import ask
history = []
MAX_HISTORY = 10
USER_ROLE = None
while True:

    question = input("\nВы: ")

    if question.lower() in {
        "exit"
    }:
        print("До свидания!")
        break

    if not question.strip():
        continue

    try:

        stream, sources = ask(
            question,
            history,
            user_role=USER_ROLE
        )

        print("\nВаш ассистент:")

        answer_parts = []

        for chunk in stream:

            text = chunk.message.content

            if text:
                print(
                    text,
                    end="",
                    flush=True
                )

                answer_parts.append(text)

        answer = "".join(answer_parts)

        print()

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        history = history[-MAX_HISTORY:]

    except Exception as e:

        print(
            f"\nОшибка: {e}"
        )
