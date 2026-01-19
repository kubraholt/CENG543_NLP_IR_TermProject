import ollama

print(" Connecting to Llama 3...")

response = ollama.chat(model='llama3:8b', messages=[
    {
        'role': 'user',
        'content': 'What is Information Retrieval? Explain in one sentence.',
    },
])

print("\n Response:")
print(response['message']['content'])