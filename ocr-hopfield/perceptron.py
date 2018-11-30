import tornado.ioloop
import tornado.web
import csv
import json


threshold = 0.5
learning_rate = 0.1
trained_perceptrons = {}
verbose = True

def is_row_empty(row):
    for i in row:
        if i != '':
            return False
    return True

def process_data(file = 'training_data.csv'):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = {}
        currentClass = ''
        for row in reader:
            if is_row_empty(row):
                pass
            else:
                if row[0] != '': #new class
                    currentClass = row[0]
                    data[currentClass] = []
                elements = map(lambda x: int(x), row[1:])    
                data[currentClass].extend(elements)
    return data

def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))

def train(training_set):
    weights = [0] * len(training_set[0][0])
    while True:
        error_count = 0
        for input_vector, desired_output in training_set:
            result = dot_product(input_vector, weights) > threshold
            error = desired_output - result
            if error != 0:
                error_count += 1
                for index, value in enumerate(input_vector):
                    weights[index] += learning_rate * error * value
        if error_count == 0:
            break
    return weights

def create_training_set(tag, training_data):
    training_set = []
    for key, data in training_data.iteritems():
        training_set.append((tuple(data),1 if key == tag else 0))
    return training_set

def classify(sensor_data, perceptrons):
    ratings = []
    for key, value in perceptrons.iteritems():
        verbose("Trying {0}".format(key))
        ratings.append((key,recognize(sensor_data,value)))
    verbose("-"*10)
    return max(ratings, key = lambda i: i[1])[0]

def recognize(sensor_data, weights):
    result = dot_product(sensor_data, weights)
    verbose("output: {0} threshold: {1}".format(result, threshold))
    return result 


def create_perceptrons():
    data = process_data()
    for tag in data.iterkeys():
        trained_perceptrons[tag] = train(create_training_set(tag, data))

def verbose(s):
    if verbose:
        print (s)

def init():
    print ("***Starting server***")
    print ("Training perceptrons...")
    create_perceptrons()
    print ("Perceptrons trained!")


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')
        
class RecognizeHandler(tornado.web.RequestHandler):
    def post(self):
        sensor_data = json.loads(self.request.body)['sensor']
        result = {'result':classify(sensor_data, trained_perceptrons)}
        self.write(result)

def make_app():
    return tornado.web.Application([
        (r"/recognize", RecognizeHandler),
        (r"/", MainHandler),
        (r"/(.*)", 
            tornado.web.StaticFileHandler,
            {"path":r"web/"})
    ])

if __name__ == "__main__":
    init()
    app = make_app()
    app.listen(8000)
    print ("Listening to port :8000")
    tornado.ioloop.IOLoop.current().start()
        
