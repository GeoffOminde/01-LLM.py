# 02-LLM.py
# LangChain + Chain-of-Thought Example

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Example Chain-of-Thought style prompt
template = """You are a reasoning assistant.
Question: {question}
Think step by step before giving the final answer."""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

# Demo run
question = "If a farmer has 17 sheep and all but 9 run away, how many are left?"
response = chain.invoke({"question": question})

print("🤔 Question:", question)
print("💡 Answer:", response.content)
