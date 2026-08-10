from utils.llm import EnterpriseAssistant


def main():
    assistant = EnterpriseAssistant()

    print("=" * 60)
    print("Enterprise AI Assistant")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You : ")

        if question.lower() == "exit":
            print("\nGoodbye.")
            break

        try:

            answer = assistant.ask(question)

            print("\nAssistant:")
            print(answer)
            print()

        except Exception as ex:

            print("\nError:")
            print(ex)
            print()


if __name__ == "__main__":
    main()
