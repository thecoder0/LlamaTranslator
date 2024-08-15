# LlamaTranslator

#### This is a system designed to translate Java Language Specification (JLS) sentences into Prolog code using the Llama3 as our language model and sentence-transformers/all-MiniLM-L6-v2 for generating embeddings. 
#### The process flow is as follows:
![Diagram](https://github.com/user-attachments/assets/a2cb0685-1b83-4666-bde9-084e77e24057)

#### 1.) Receive the JLS sentence as our query
#### 2.) Generate query embedding using the embedding model
#### 3.) Perform Hybrid Search on the Knowledge Base
#### 4.) Use RAG to retrieve relevant context and augment prompts
#### 5.) Use APE for generating the prompt for expanding the query
#### 6.) Feed to LLM to process the prompt to generate the expanded query
#### 7.) The LlamaTranslator system generates a translation prompt using the expanded query and context from RAG
#### 8.) The LLM processes the translation prompt to generate Prolog translation and identify intersecting predicates
#### 9.) The knowledge base is updated and returns the information on translation with intersecting predicates

#### Planned improvements: 
#### • Add evaluation metrics (TBD) for improving consistency and accuracy of JLS-to-Prolog translations
#### • Implement feedback mechanism for rating translations and for improving APE and RAG processes
#### • Incorporate cross-reference solution by aggregating information from referenced section in the JLS to the original statement
