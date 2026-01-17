import chromadb
#lient = chromadb.PersistentClient(path='data/chroma_db')
# = client.get_collection('embeddings_t1_LSX_UniWue_ModernGBERT_1B')
#rint(f'Count: {c.count()}')



client = chromadb.PersistentClient(path='data/chroma_db')
for c in client.list_collections():
	try:
		count = c.count()
		print(f'{c.name}: {count:,}')
	except Exception as e:
		print(f'{c.name}: ERROR - {e}')