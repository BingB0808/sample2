import flask
import pickle
import numpy 

# Use pickle to load in the pre-trained model.
#fn = 'model/GA1.pkl'
fn = 'irisReg-model.pkl'

model_instance = pickle.load(open(fn,'rb'))

app = flask.Flask(__name__, template_folder='pages')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('home.html'))
    if flask.request.method == 'POST':
        sepal_lenght = flask.request.form['sepal_lenght']
        sepal_width = flask.request.form['sepal_width']
        petal_lenght = flask.request.form['petal_lenght']
        petal_width = flask.request.form['petal_width']
        
        input_variables = numpy.array([[ sepal_lenght,sepal_width,petal_lenght,petal_width]])

        array_inputs =  input_variables.astype(numpy.float)
        
        predictions = model_instance.predict(array_inputs)
        predictions = str(predictions).strip('[]')
        
        #predictions = pygad.nn.predict(last_layer=model_instance,data_inputs=array_inputs)
        print(predictions)
        
        return flask.render_template('home.html',
                                     original_input={'sepal_lenght': sepal_lenght,
                                                     'sepal_width':sepal_width,
                                                     'petal_lenght':petal_lenght,
                                                     'petal_width':petal_width},
                                     result=str(predictions)) 

if __name__ == '__main__':
    app.run()