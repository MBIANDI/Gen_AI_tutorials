from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings


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