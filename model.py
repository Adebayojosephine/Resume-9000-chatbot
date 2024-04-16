from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/'

custom_prompt_template = """Provide information about resume qualifications, skills, and experience in response to the user's question.

Context: {context}
Question: {question}

Return a relevant answer.
Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 4098,
        temperature = 0.5
    )
    return llm


# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)
#     return qa

def qa_bot():
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Load the language model
    llm = load_llm()
    
    # Set the custom prompt
    qa_prompt = set_custom_prompt()
    
    # Initialize the retrieval QA chain with refined settings
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa_chain


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

############
# Chainlit #
############
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Firing up the bot...")
    await msg.send()
    msg.content = "**Welcome to ResumeBot 9000!** Crafted by Josephine Adebayo, it's your personalized resume assistant for creating professional resumes effortlessly."
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()

