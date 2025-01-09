from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain.docstore.document import Document as LangchainDocument
from typing import List, Tuple, Optional
from tqdm import tqdm
import json
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
import os


def instanciate_llm_with_huggingface(
        model_name: str,  
        max_new_tokens: int,
        do_sample: bool, 
        temperature: float, 
        top_p: float, 
        repetition_penalty: float
        ) -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        
    )

def initialize_embeddings_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    embeddings_model = HuggingFaceEmbeddings(model_name = model_name, encode_kwargs={
            "normalize_embeddings": True
        })
    return embeddings_model

def answer_with_rag(
    question: str,
    rag_prompt_template: str,
    llm: LLM,
    knowledge_index: VectorStore,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
    
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    final_prompt = rag_prompt_template.format(question=question, context=context)

    # Redact an answer
    answer = llm.invoke(final_prompt)

    return answer, relevant_docs

from langchain_core.language_models import BaseChatModel

def run_rag_tests(
    eval_dataset,
    llm,
    rag_prompt_template,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for id, example in tqdm(eval_dataset.iterrows()):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        
        answer, relevant_docs = answer_with_rag(
            question = question,
            llm = llm,
            knowledge_index = knowledge_index,
            reranker=reranker,
            rag_prompt_template = rag_prompt_template
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)

def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
             item.strip() for item in eval_result.content.split("[RESULT]")
         ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)