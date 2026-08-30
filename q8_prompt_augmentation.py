from langchain_core.prompts import ChatPromptTemplate

# RAG Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
Answer the question only using the provided context.
If the answer is not found in the context, say:
"I could not find the answer."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

# Example retrieved context
context = """
TechNova Solutions was founded in 2020.
The company provides software development and AI services.
Its headquarters is located in Pune, Maharashtra.
"""

question = "When was TechNova Solutions founded?"

# Pass context and question into the prompt
final_prompt = prompt.invoke({
    "context": context,
    "question": question
})

# Display the prompt
print("Augmented Prompt:")
print(final_prompt)