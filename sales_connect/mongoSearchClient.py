import pymongo
from hashlib import md5
from time import time


class Mongo(object):
    def __init__(self, clientURL, dbName):
        self.clientURL = clientURL
        self.dbName = dbName
        self.conn = None
        self.db = None
        self.coll = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, type, value, traceback):
        self.disconnect()

    def connect(self):
        self.conn = pymongo.MongoClient(self.clientURL)
        self.db = self.conn[self.dbName]
        return self

    def disconnect(self):
        self.conn = self.conn.close()
        self.conn = None
        self.db = None
        self.coll = None

    def setCollection(self, collName):
        self.coll = self.db[collName]

    def setDB(self, dbName):
        self.db = self.conn[dbName]

    def renameCollection(self, newName):
        self.coll = self.rename(newName)


class MongoSalesConnect(Mongo):

    def inferWrite(self, url, payload, inference):
        mongoData = {'_id': self.hashMD5(url),
                     'payload': payload,
                     'recommendation': inference,
                     'url': url,
                     'timestamp': int(time())}
        self.coll.update({"_id": self.hashMD5(url)},
                         mongoData, upsert=True)

    def feedbackUpdate(self, url, goldenStandard):
        mongoData = {"$set": {'goldenStandard': goldenStandard,
                              'timestamp': int(time())}}
        query = {'_id': self.hashMD5(url)}
        self.coll.update_many(query, mongoData)

    def hashMD5(self, text):
        return md5(text.encode()).hexdigest()
