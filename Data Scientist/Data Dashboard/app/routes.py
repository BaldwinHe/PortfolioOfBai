from app import app
import json, plotly
from flask import render_template
from data.data_wrangling.fetchdata import return_figure

@app.route('/')
@app.route('/index')
def index():
    figures = return_figure()

    ids = ['figure-{}'.format(i) for i,_ in enumerate(figures)]

    figuresJSON = json.dumps(figures,cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',ids=ids,figuresJSON=figuresJSON)