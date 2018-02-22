import spacy
import json

data = json.load(open('./data.json'))

nlp = spacy.load('en')

for i in range (0, len(data)):
    print("Lyrics : " + data[i]["lyrics"])
    print("")
    lyrics = nlp(data[i]["lyrics"])
    for ent in lyrics.ents:
        print(ent.text, ent.label_)
    print("----------------------------------------")
    
