from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from collections import Counter
from string import punctuation
from heapq import nlargest

app = FastAPI()

# Configuração do CORS
origins = [
    "http://localhost:3000",  # Permitir o frontend React local
    "http://localhost",       # Permitir localhost
    "http://localhost:5173",  # Caso esteja rodando no Vite
    "https://kauebrandao.github.io",  # Permitir o domínio do seu site no GitHub Pages
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir origens específicas
    allow_credentials=True,
    allow_methods=["*"],    # Permitir qualquer método (GET, POST, etc.)
    allow_headers=["*"],    # Permitir qualquer cabeçalho
)

# Modelo de entrada para a requisição
class TextInput(BaseModel):
    text: str
    num_sentences: int

def extract_keywords(text: str, num_keywords: int = 5):
    # Carrega o modelo de linguagem do spaCy para português
    nlp = spacy.load("pt_core_news_sm")
    
    # Processa o texto
    doc = nlp(text)
    
    # Extrai palavras, ignorando verbos, conjunções, stop words e pontuação
    keywords = []
    for token in doc:
        # Considera apenas substantivos e adjetivos, excluindo stop words e pontuação
        if (token.pos_ in ["NOUN", "ADJ"]) and not token.is_stop and not token.is_punct and token.text.strip():
            keywords.append(token.text.lower())
    
    # Calcula a frequência das palavras
    keyword_frequencies = Counter(keywords)
    
    # Seleciona as palavras mais frequentes
    top_keywords = [word for word, _ in keyword_frequencies.most_common(num_keywords)]
    
    return top_keywords if top_keywords else ["Nenhuma palavra-chave encontrada."]

def summarize_text(text: str, num_sentences: int):
    # Carrega o modelo de linguagem do spaCy para português
    nlp = spacy.load("pt_core_news_sm")
    
    # Processa o texto
    doc = nlp(text)
    
    # Extrai as sentenças, garantindo que sejam válidas
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return "Texto vazio ou sem sentenças válidas.", []
    
    # Limita o número de sentenças ao total disponível
    num_sentences = min(num_sentences, len(sentences))
    
    # Calcula a frequência das palavras (ignorando stop words e pontuação)
    word_frequencies = Counter()
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip():
            word_frequencies[token.text.lower()] += 1
    
    # Palavras-chave para dar peso extra a temas importantes
    key_themes = ["identidade", "redes", "sociais", "pertencimento", "autenticidade", 
                  "saúde", "mental", "planejamento", "urbano", "políticas", "públicas", "cidade"]
    
    # Atribui um peso para cada sentença
    sentence_scores = {}
    for i, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if sent_text:  # Ignora sentenças vazias
            # Calcula o escore com base nas palavras frequentes
            score = sum(word_frequencies.get(token.text.lower(), 0) 
                        for token in sent 
                        if not token.is_stop and not token.is_punct and token.text.strip())
            # Aumenta o peso para palavras-chave
            for theme in key_themes:
                if theme in sent_text.lower():
                    score += 5  # Peso extra para temas centrais
            # Aumenta o peso para a primeira e última sentença
            if i == 0 or i == len(sentences) - 1:
                score += 3  # Peso extra para introdução e conclusão
            sentence_scores[sent_text] = score
    
    # Seleciona as sentenças com maior pontuação
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Mantém a ordem original das sentenças
    summary = [sent for sent in sentences if sent in summary_sentences]
    
    return " ".join(summary), summary_sentences

@app.post("/summarize/")
async def summarize(input_data: TextInput):
    # Valida a entrada
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode estar vazio.")
    if input_data.num_sentences <= 0:
        raise HTTPException(status_code=400, detail="O número de sentenças deve ser positivo.")
    
    try:
        # Gera o resumo e extrai palavras-chave
        summary, _ = summarize_text(input_data.text, input_data.num_sentences)
        keywords = extract_keywords(input_data.text, num_keywords=5)
        
        # Retorna a resposta em formato JSON
        return {
            "summary": summary,
            "keywords": keywords
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o texto: {str(e)}")
