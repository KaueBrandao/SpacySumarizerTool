from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from heapq import nlargest
import uvicorn
import os

# Baixar o modelo do spaCy no startup
os.system("python -m spacy download pt_core_news_sm")

# Carregar modelo de NLP do spaCy
nlp = spacy.load("pt_core_news_sm")

# Inicializar FastAPI
app = FastAPI()

# Configuração do CORS
origins = [
    "http://localhost:3000",  # Permitir o frontend React local
    "http://localhost",  # Permitir localhost
    "http://localhost:5173",  # Caso esteja rodando no Vite
    "https://kauebrandao.github.io",  # Permitir o domínio do seu site no GitHub Pages
    # Adicione outros domínios conforme necessário
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir origens específicas
    allow_credentials=True,
    allow_methods=["*"],  # Permitir qualquer método (GET, POST, etc.)
    allow_headers=["*"],  # Permitir qualquer cabeçalho
)

class TextoEntrada(BaseModel):
    texto: str
    limite_palavras: int = 100  # Valor padrão de 100 palavras

def resumir_texto_spacy(texto, limite_palavras):
    doc = nlp(texto)

    palavras_importantes = {}
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip():
            palavra = token.lemma_.lower()
            palavras_importantes[palavra] = palavras_importantes.get(palavra, 0) + 1

    # Obter as 5 palavras mais frequentes
    palavras_frequentes = nlargest(5, palavras_importantes, key=palavras_importantes.get)

    # Criar o resumo com frases contendo palavras-chave
    sentencas = [sent.text for sent in doc.sents if any(palavra in sent.text.lower() for palavra in palavras_frequentes)]

    # Limitar o resumo a um número máximo de palavras
    resumo = []
    total_palavras = 0
    for sentenca in sentencas:
        palavras_sentenca = sentenca.split()
        if total_palavras + len(palavras_sentenca) <= limite_palavras:
            resumo.append(sentenca)
            total_palavras += len(palavras_sentenca)
        else:
            break  # Interrompe quando o limite de palavras for atingido

    return ' '.join(resumo), palavras_frequentes

@app.post("/resumir")
async def resumir_texto(dados: TextoEntrada):
    if not dados.texto.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio.")

    resumo, palavras_chave = resumir_texto_spacy(dados.texto, dados.limite_palavras)
    return {"resumo": resumo, "palavras_chave": palavras_chave}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)