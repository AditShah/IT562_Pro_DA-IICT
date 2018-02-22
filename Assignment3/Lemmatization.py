import spacy
import json

data = json.load(open('./data.json'))

nlp = spacy.load('en')

for i in range (0, len(data)):
    print("Title : " + data[i]["title"])
    print("")
    title = nlp(data[i]["title"])
    for word in title:
        print(word.text, word.lemma_)
    print("\n\n\n")
    print("Artist : " + data[i]["artist"])
    print("")
    artist = nlp(data[i]["artist"])
    for word in artist:
        print(word.text, word.lemma_)
    print("-----------------------------------------------")
