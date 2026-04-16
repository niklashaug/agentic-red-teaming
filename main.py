from ollama import chat
from ollama import ChatResponse


def main():
    print("Hello from agentic-red-teaming!")
    response: ChatResponse = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": "Why is the sky blue?",
            },
        ],
    )
    print(response["message"]["content"])
    # or access fields directly from the response object
    print(response.message.content)


if __name__ == "__main__":
    main()
