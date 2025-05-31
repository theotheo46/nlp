Переработай функцию fill_vector_base так, чтобы заполнение vector_store проводилось батчами размером batch_size и цикл использовал tqdm

<fill_vector_base>
def fill_vector_base():
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",  # Where to save data locally, remove if not neccesary
    )

    documents = [Document(page_content=article) for article in articles[:100]]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
</fill_vector_base>
