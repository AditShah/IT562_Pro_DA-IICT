import spacy
import json

data = json.load(open('./data.json'))

nlp = spacy.load('en')

for i in range (0, len(data)):
    print("Lyrics : " + data[i]["lyrics"])
    print("")
    lyrics = nlp(data[i]["lyrics"])
    print("Noun Chunk : ")
    for noun in lyrics.noun_chunks:
        print(noun.root.text)
    print("")
    print("Artist : " + data[i]["artist"])
    print("")
    artist = nlp(data[i]["artist"])
    print("Noun Chunk : ")
    for noun in artist.noun_chunks:
        print(noun.root.text)
    print("-----------------------------------------------")
    
