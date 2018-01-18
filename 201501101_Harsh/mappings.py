import json
import requests
from elasticsearch import Elasticsearch
from collections import namedtuple

es = Elasticsearch([{
        'host': 'localhost',
        'port': 9200
    }])

request_body = {
    "mappings": {
        "song": {
            "properties": {
                "artist": {"type": "text"},
                "song_length": {"type": "float"},
                "genre": {"type": "text"},
                "title": {"type": "text"},
                "id": {"type": "long"},
                "release_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                }
            }
        }
    },
    
    "settings" : {
        "number_of_shards": 5,
        "number_of_replicas": 1
    }
}

es.indices.create(index = 'music', body = request_body)

data = json.loads(open("./data.json").read())

i = 0

while i < 500:
    es.index(index = 'music', doc_type = 'song', id = i, body = data[i])
    i += 1
