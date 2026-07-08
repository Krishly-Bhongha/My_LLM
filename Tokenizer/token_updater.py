import pickle as pkl
with open('tokenizer/converter.pkl', 'rb') as f:
    converter = pkl.load(f)
for key in converter["id_to_token"]:
    print(key, converter["id_to_token"][key])

