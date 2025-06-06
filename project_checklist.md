## ✅ Fase 1: Núcleo funcional mínimo (MVP RAG)

* [x] Implementar y testear `summarize_html_with_ai` (web agent)
* [x] Implementar y testear `chunk_text` (chunker)
* [x] Crear `embedder` para convertir chunks en vectores
* [x] crear `indexer`
* [ ] Construir `retriever` vectorial (ej. FAISS, Chroma, etc.)
* [ ] Implementar `generate_answer(query, context_chunks)` con LLM
* [ ] Crear script de prueba end-to-end (query → answer)

## 🔄 Fase 2: Testing y validación

* [x] Prueba unitaria para chunker con texto real
* [ ] Agregar test de borde: texto corto, sin puntos, vacío
* [ ] Agregar tests para embedder y retriever
* [ ] Validar respuestas con LLM en distintos casos
* [ ] Test end-to-end del pipeline completo (input → respuesta)

## 🧱 Fase 3: Estructura y arquitectura

* [x] Crear estructura básica del proyecto (`rag_app/`)
* [ ] Separar claramente `agents/`, `ingestion/`, `retriever/`, `generator/`, `api/`
* [ ] Documentar funciones y módulos con docstrings
* [ ] Incluir logger profesional (opcional para depuración)

## 🐳 Fase 4: Empaquetado y despliegue

* [ ] Crear `pyproject.toml` limpio y reproducible
* [ ] Crear `Dockerfile` optimizado (basado en Python slim o similar)
* [ ] Crear `.dockerignore`
* [ ] Agregar `entrypoint.sh` (si es útil para inicialización)
* [ ] Probar build de Docker e iniciar contenedor correctamente
* [ ] Probar contenedor desde otra máquina (usuario final sin dev tools)

## 📚 Fase 5: Documentación y entrega

* [ ] Crear `README.md` con instrucciones claras de uso
* [ ] Agregar ejemplos de uso de la API o CLI
* [ ] Documentar los pasos para ejecutar en Docker
* [ ] Incluir información de contacto, licencia y autoría (si aplica)
