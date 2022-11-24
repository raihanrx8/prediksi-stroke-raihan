import sklearn
from flask import Flask, render_template, request
from model import load, prediksi

app = Flask(__name__)

# load model dan scaler
load()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # menangkap data yang diinput user melalui form
    jenis_kelamin = int(request.form['jenis_kelamin'])
    umur = int(request.form['umur'])
    hipertensi = int(request.form['hipertensi'])
    penyakit_jantung = int(request.form['penyakit_jantung'])
    
    pernah_menikah = int(request.form['pernah_menikah'])
    tipe_pekerjaan = int(request.form['tipe_pekerjaan'])
    tipe_rumah = int(request.form['tipe_rumah'])
    rata_rata_level_glukosa = request.form['rata_rata_level_glukosa']
    berat_badan = int(request.form['berat_badan'])
    tinggi_badan = int(request.form['tinggi_badan'])
    bmi= berat_badan/(tinggi_badan*tinggi_badan)
    status_merokok = int(request.form['status_merokok'])
    

    # melakukan prediksi menggunakan model yang telah dibuat
    data = [[jenis_kelamin, umur, hipertensi, penyakit_jantung, pernah_menikah, 
                    tipe_pekerjaan, tipe_rumah, rata_rata_level_glukosa, bmi, status_merokok]]
    
    prediction_result,confidence= prediksi(data)
    return render_template('index.html', hasil_prediksi=prediction_result,nilai_kepercayaan=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)