from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response

import os

import pandas

from catboost import CatBoostClassifier

d=pandas.read_csv("diabetes_data_upload (1).csv")
print(len(d.iloc[0]))
def hello_world(request):
    print('Incoming request')
    return Response('Hello')

model=CatBoostClassifier(cat_features=list(d.columns))
print(list(d.columns))
model.load_model("Diabetes (1).cbm")

y=pd.DataFrame()
y["class"]=d["class"]

d.pop("class")
y["class"], _=y["class"].factorize()
y["class"]=1-y["class"]
explainer = ClassifierExplainer(
                model, d, y,
                labels=["Negative","Positive"]
                )

db = ExplainerDashboard(explainer, title="Diabetes Explainer",
                    shap_dependence=False,
                    shap_interaction=False,
                    decision_trees=False)
def dashboard(request):
    return db.app.index()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 2000))
    
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        config.add_route('predict', '/predict')
        config.add_view(predict, route_name='predict')
        app = config.make_wsgi_app()
    serve(app, host='0.0.0.0', port=port)
