from flask_restful import Resource, reqparse
from sklearn.model_selection import train_test_split
import uuid
import sqlite3
import numpy as np
from NaiveBayes import NaiveBayes


def initiate_nb():
    conn = sqlite3.connect("FakeNews.sqlite")
    c = conn.cursor()
    c.execute("Select * from TrainingData")
    results = c.fetchall()
    x_train = [row[2] for row in results]
    y_train = [row[1] for row in results]
    conn.close()

    train_data, test_data, train_labels, test_labels = train_test_split(x_train, y_train, shuffle=True, test_size=0.25,
                                                                        random_state=42, stratify=y_train)
    classes = np.unique(train_labels)

    # Training phase....
    nb = NaiveBayes(classes)
    nb.train(train_data, train_labels)
    return nb


def addResult(result):
    conn = sqlite3.connect("FakeNews.sqlite")
    c = conn.cursor()
    c.execute("INSERT INTO TrainingData VALUES(?, ?, ?)",
              (str(uuid.uuid4()), result["predictionCode"], result["text"]))
    conn.commit()
    conn.close()


class NBPredict(Resource):
    nb = initiate_nb()
    count = 0

    def put(self):
        parser = reqparse.RequestParser()
        parser.add_argument("text")
        args = parser.parse_args()
        input_text = args["text"]
        prediction = self.nb.test([input_text])
        prediction_text = "Positive" if prediction[0] == 1 else "Negative"
        result = {
            "predictionCode": int(prediction[0]),
            "text": input_text,
            "predictionText": prediction_text
        }
        addResult(result)
        self.count = self.count + 1
        if self.count > 5:
            self.nb = initiate_nb()
            print("hello")
        return result, 201
