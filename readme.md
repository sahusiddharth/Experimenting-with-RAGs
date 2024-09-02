# Introduction

Inspired by a research paper(refrenced below) on increasing the alignment between user queries and data, I decided to create a code that prepares your data for Query-Based Retrieval Augmented Generation (QB-RAG). I utilized Langchain classes for the LLM and vector store, and I incorporated Pydantic to gain better control over the responses from our LLMs.

In this README, I'll start with a brief overview of the paper based on my understanding, followed by a setup guide if you'd like to use the code. I'll also share some tips on how to make the most of this technique in your own projects. Finally, I'll conclude with my key takeaways.

## Query-Based Retrieval Augmented Generation (QB-RAG)
This paper introduces QB-RAG, a novel approach for enhancing the retrieval phase of RAG systems. QB-RAG addresses the inherent semantic misalignment between user queries and knowledge bases by pre-generating a comprehensive set of potential questions directly from the content. We then leverage efficient vector search to map incoming online queries to these pre-computed questions, facilitating a more accurate and aligned retrieval process. While QB-RAG requires a significant offline computational investment for question generation, this trade-off is strategically advantageous. Unlike conventional methods that rely on online LLM calls for query rewriting or enhancement, QB- RAG shifts this computational burden offline. This distinction is crucial for minimizing latency in real-time applications, as online LLM invocations can significantly impact the user experience.

### Setup Instructions

To set up the QB-RAG converter, you can use any LLM and vector store offered by Langchain. I used OpenAI and Pinecone, respectively. If you'd like to follow this implementation, ensure you have the following dependencies installed. This guide assumes you're using Python 3.11. You can install the required libraries with the following commands:

```shell
pip install python-dotenv
pip install langchain-openai
pip install pinecone
pip install langchain-pinecone
```

For convenience, you can refer to the `requirements.txt` file, which lists all the dependencies required for the code to run.

To manage secret keys, I use `dotenv`. If you choose this method, store your secret keys inside the `.env` file.

### Adapting to Your Use Case

To fully harness the power of QB-RAG, consider the following adjustments:

- **Customizing Prompts:**  
  Tailor the role and task based on your specific use case. When writing one/few-shot examples, consider the breadth and depth of questions your users might ask that can be answered using the provided context or document text.

- **Controlling Question Generation:**  
  The number of questions generated per chunk should depend on the complexity of each chunk. For example, if your chunks are generally information-dense, increase the number of questions generated per chunk.

By customizing these aspects, you can optimize the technique to better suit your needs. For detailed guidance on making these adjustments, refer to the `main.ipynb` file.

### Takeaways

1. **The degree of alignment is highly dependent on the quality of questions generated** from the context. So, choose the LLM responsible for generating the questions wisely.
2. **Lack of variety in the retrieved context** can occur. For instance, if you're using the top 5 retrieved contexts based on question similarity, you might retrieve the same context multiple times, reducing variety. This often happens if more questions are generated from less information-dense contexts. This ties back to the importance of generating high-quality questions.

### References

1. **The Geometry of Queries: Query-Based Innovations in Retrieval-Augmented Generation**  
   Authors: Eric Yang, Jonathan Amar, Jong Ha Lee, Bhawesh Kumar, Yugang Jia  
   [Read the Paper](https://arxiv.org/abs/2407.18044)
