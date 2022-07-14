import base64
from PIL import Image, ImageFilter
import heartbeat as hb
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
import uvicorn
from fastapi.responses import FileResponse
from io import BytesIO
from starlette.responses import StreamingResponse


app = FastAPI()

app.mount('/data', StaticFiles(directory="data"), name="data")


        
@app.post("/analysis")
async def add_analysis(csv: UploadFile=File(...)):
    hrw = 0.75 #sampling
    fs = 100 # frequncy record
    text = secure_filename(csv.filename)
    contents = csv.file.read()
    with open(f"./data/"+text, 'wb') as f:
        f.write(contents)
        
    ecg = hb.get_data("./data/data.csv")
    
    # setup
    await hb.rolmean(ecg, hrw, fs)
    
    await hb.detect_peaks(ecg)
    
    await hb.calc_RR(ecg, fs)
    
    # time domain
    await hb.calc_ts_measures()

    # frequency domain
    await hb.calc_fs_measures()
    
    n = len(ecg.hart)
    freq = np.fft.fftfreq(len(ecg.hart), d = ((1/fs)))
    freq = freq[range(n//2)]
    
    ## Do fft
    interpolate = hb.measures['interpolate']
    Y = np.fft.fft(interpolate)/n
    Y = Y[range(n//2)]
    
    lf = np.trapz(abs(Y[(freq>=0.04) & (freq<=0.15)]))
    print("LF: ", lf)
    
    hf = np.trapz(abs(Y[(freq>=0.16) & (freq<=0.5)]))
    print("HF: ", hf)
    
    # original signal image
    plt.figure(1)
    plt.title("Sinyal Jantung")
    plt.plot(ecg.hart)
    plt.xlabel("Waktu (ms)")
    plt.ylabel("EKG (mv)")
    plt.savefig("./images/original_image.png")
    
    #save interpolate image
    RR_x = hb.measures['RR_X']
    RR_y = hb.measures['RR_Y']
    RR_x_new = hb.measures['RR_x_new']
    plt.figure(2)
    plt.title("Puncak Sinyal Asli dan Interpolasi")
    plt.plot(RR_x, RR_y, label="Original", color='blue')
    plt.plot(RR_x_new, interpolate, label="Interpolated", color='red')
    plt.xlabel("Waktu (ms)")
    plt.ylabel("EKG (mv)")
    plt.legend()
    plt.savefig('./images/interpolate.png')
    
    # Frekuensi spektrum image
    plt.figure(3)
    plt.title("Frekuensi Spektrum Sinyal Jantung")
    plt.xlim(0,0.6)
    plt.ylim(0, 50)
    plt.plot(freq, abs(Y))
    plt.xlabel("Frekuensi (Hz)")
    plt.savefig("./images/spectrum.png")
    
   # fill json data
    ibi = hb.measures['ibi']
    sdnn = hb.measures['sdnn']
    sdsd = hb.measures['sdsd']
    rmssd = hb.measures['rmssd']
    bpm = hb.measures['bpm']
    
    JSON = {
        'ibi': ibi,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'rmssd': rmssd,
        'bpm': bpm,
        'lf': lf,
        'hf': hf
    }
        
    response = {
        'data': JSON
    }
    
    return response

@app.get("/original")
async def original_signal():
    file_path = "./images/original_image.png"
    with open(file_path, "rb") as image_file:
        encode_image = base64.b64encode(image_file.read())
    return encode_image
    

@app.get("/interpolate")
async def interpolate():
    file_path = "./images/interpolate.png"
    with open(file_path, "rb") as image_file:
        encode_image = base64.b64encode(image_file.read())
    return encode_image
    

@app.get("/spectrum")
async def spectrum():
    file_path = "./images/spectrum.png"
    with open(file_path, "rb") as image_file:
        encode_image = base64.b64encode(image_file.read())
    return encode_image
        

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)