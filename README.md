# ragapp
Kleines RAG Projekt mit Flask + Chroma

Dieses Projekt ist ein einfacher RAG-Prototyp (Retrieval-Augmented Generation):
PDFs werden hochgeladen, in Chunks zerlegt, als Embeddings in Chroma gespeichert und anschließend als Kontext für Chat-Antworten verwendet.

## Features

PDF Upload via Flask

Chunking (CharacterTextSplitter)

Speicherung in Chroma Vector DB (persistiert)

Fragebeantwortung per OpenAI Chat Completion im JSON-Format (basierend auf Context)

## Requirements

Python 3.10+

OpenAI API Key als Environment Variable