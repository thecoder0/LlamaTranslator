import requests
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re
from collections import Counter

class LlamaTranslator:
    def __init__(self, llm_model="llama3:latest", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.llm_model = llm_model
        self.ollama_url = "http://localhost:11434/api/generate"

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

        self.knowledge_base = []

    # Adds a new entry to the knowledge base
    def add_to_knowledge_base(self, jls_sentence, prolog_code):
        embedding = self.get_embedding(jls_sentence)
        keywords = self.extract_keywords(jls_sentence)

        self.knowledge_base.append({
            "jls_sentence": jls_sentence,
            "prolog_code": prolog_code,
            "embedding": embedding,
            "keywords": keywords
        })

    # Sends a prompt to the ollama API and returns a generated response
    def ollama_generate(self, prompt):
        data = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.ollama_url, json=data)
        if response.status_code == 200:
            return json.loads(response.text)['response']
        else:
            raise Exception(f"Error from Ollama API: {response.text}")
    
    # Generates an embedding for the input text
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Calculates the cosine similarity between two vectors
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

     # Extracts 5 most common words and returns the filtered list of words that are longer than 3 characters
    def extract_keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        return [word for word, _ in Counter(words).most_common(5) if len(word) > 3]

    def keyword_similarity(self, keywords1, keywords2):
        set1 = set(keywords1)
        set2 = set(keywords2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    # Performs a hybrid search on the knowledge base using both semantic and keyword similarity,
    # and returns a list of top k most similar entries
    def hybrid_search(self, query, k=3, semantic_weight=0.7):
        if not self.knowledge_base:
            return []

        query_embedding = self.get_embedding(query)
        query_keywords = self.extract_keywords(query)

        similarities = []
        for item in self.knowledge_base:
            semantic_sim = self.cosine_similarity(query_embedding, item['embedding'])
            keyword_sim = self.keyword_similarity(query_keywords, item['keywords'])
            combined_sim = semantic_weight * semantic_sim + (1 - semantic_weight) * keyword_sim
            similarities.append(combined_sim)

        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
        return [self.knowledge_base[i] for i in top_indices]

    # Uses APE to expand the original query by retrieving relevant context from the knowledge base and adding it
    def ape_expand_query(self, query):
        relevant_context = self.hybrid_search(query, k=3)
        context_str = "\n".join([f"Related JLS: {item['jls_sentence']}" for item in relevant_context])
        
        prompt = f"""
        Given the following Java Language Specification (JLS) sentence and related context, expand the original sentence to include more relevant details from the context. Do not rephrase, just expand. If the context is not directly relevant or insufficient, return the original sentence unchanged.

        Original: "{query}"

        Context:
        {context_str}

        Expanded JLS sentence:
        """
        
        response = self.ollama_generate(prompt)
        return response.strip()

    def extract_predicates(self, prolog_code):
        return set(re.findall(r'(\w+)\(', prolog_code))

    def find_intersecting_predicates(self, new_predicates, existing_predicates):
        return new_predicates.intersection(existing_predicates)

    def generate_prompt(self, jls_sentence, context):
        context_str = "\n".join([f"JLS: {item['jls_sentence']}\nProlog: {item['prolog_code']}" 
                                 for item in context])
        return f"""
        Translate the following Java Language Specification (JLS) sentence to Prolog code without any explanation.
        Use these examples as reference only if directly relevant: {context_str}
        Now, translate this JLS sentence to Prolog without any explanation: "{jls_sentence}" 
        Prolog translation: """

    # Translates the given JLS sentence into Prolog
    def translate_to_prolog(self, jls_sentence):
        expanded_query = self.ape_expand_query(jls_sentence)
        relevant_context = self.hybrid_search(expanded_query, k=2)
        prompt = self.generate_prompt(expanded_query, relevant_context)
        translation = self.ollama_generate(prompt)
        
        new_predicates = self.extract_predicates(translation)
        
        # Loop to identify intersecting predicates with existing knowledge base entries
        for item in self.knowledge_base:
            existing_predicates = self.extract_predicates(item['prolog_code'])
            intersecting_predicates = self.find_intersecting_predicates(new_predicates, existing_predicates)
            
            if intersecting_predicates:
                print(f"Found intersecting predicates: {intersecting_predicates}")
                print(f"Related context: {item['jls_sentence']}")
        
        self.add_to_knowledge_base(jls_sentence, translation)
        
        return translation

translator = LlamaTranslator()

print(f"***************************************")

jls_sentence1 = "The scope of a declaration is the region of the program within which the entity declared by the declaration can be referred to using a simple name, provided it is not shadowed."
prolog_translation1 = translator.translate_to_prolog(jls_sentence1)
print(f"JLS: {jls_sentence1}")
print(f"\nExpanded: {translator.ape_expand_query(jls_sentence1)}")
print(f"\nProlog: \n{prolog_translation1}")

print(f"***************************************")

jls_sentence2 = "The scope of a class's type parameter is the type parameter section of the class declaration, and the type parameter section of any superclass type or superinterface type of the class declaration, and the class body."
prolog_translation2 = translator.translate_to_prolog(jls_sentence2)
print(f"JLS: {jls_sentence2}")
print(f"\nExpanded: {translator.ape_expand_query(jls_sentence2)}")
print(f"\nProlog: \n{prolog_translation2}")

print(f"***************************************")

jls_sentence3 = "A local variable, formal parameter, exception parameter, local class, or local interface can only be referred to using a simple name, not a qualified name."
prolog_translation3 = translator.translate_to_prolog(jls_sentence3)
print(f"JLS: {jls_sentence3}")
print(f"\nExpanded: {translator.ape_expand_query(jls_sentence3)}")
print(f"\nProlog: \n{prolog_translation3}")