from flask import Flask, flash, request, redirect, url_for, render_template
from flask import session
import pickle



import numpy as np



 
app = Flask(__name__)

@app.route('/') 
def home():
    return render_template('first.html')
 
@app.route('/text',methods=['POST'])
def text():
    
    int_features = [float(x)for x in request.form.values()]
    final_features = [np.array(int_features)]
    model=pickle.load(open('saved_metal_model','rb'))
    ans=model.predict(final_features)
    #return render_template("first.html",prediction=ans)
    for i in range(0,len(ans)):
        if ans[i]==0:
            return render_template("first.html",prediction="No Defect")
        if ans[i]==1:
            return render_template("first.html",prediction="Tungsten inclusion")
        if ans[i]==2:
            return render_template("first.html",prediction="Porosity")
        

        
            
    

    
    
    
    
    
 
if __name__ == "__main__":
    app.run(debug=True)

