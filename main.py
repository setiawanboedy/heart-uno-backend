from PIL import Image, ImageFilter
import heartbeat as hb
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
import uvicorn
from io import BytesIO
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse


app = FastAPI()

app.mount('/data', StaticFiles(directory="data"), name="data")


        
@app.post("/analysis")
async def add_analysis(file:UploadFile=File(...)):
    sample_rate = 360
    hrw = 0.75 #sampling
    fs = 100 # frequncy record
    text = secure_filename(file.filename)
    contents = file.file.read()
    with open(f"./data/"+text, 'wb') as f:
        f.write(contents)
        
    ecg = hb.get_data("./data/data.csv")
    
    # setup
    hb.rolmean(ecg, hrw, fs)
    
    hb.detect_peaks(ecg)
    
    hb.calc_RR(ecg, fs)
    
    # time domain
    hb.calc_ts_measures()

    # frequency domain
    n = len(ecg.hart)
    freq = np.fft.fftshift(len(ecg.hart), d = ((1/fs)))
    freq = freq[range(n//2)]
    
    ## Do fft
    interpolate = hb.measures['interpolate']
    Y = np.fft.fft(interpolate)/n
    Y = Y[range(n//2)]
    
    lf = np.trapz(abs(Y[(freq>=0.04) & (freq<=0.15)]))
    print("LF: ", lf)
    
    hf = np.trapz(abs(Y[(freq>=0.16) & (freq<=0.5)]))
    print("HF: ", hf)
    
   # fill json data
    ibi = hb.measures['ibi']
    sdnn = hb.measures['sdnn']
    sdsd = hb.measures['sdsd']
    rmssd = hb.measures['rmssd']
    
    JSON = {
        'ibi': ibi,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'rmssd': rmssd
    }
        
    response = {
        'data': JSON
    }
    
    return response

@app.get("/image_signal")
async def image_signal():
    file_path = "./images/image_signal.PNG"
    original_image = Image.open(file_path)
    original_image = original_image.filter(ImageFilter.BLUR)

    filtered_image = BytesIO()
    original_image.save(filtered_image, "PNG")
    filtered_image.seek(0)
    return StreamingResponse(filtered_image, media_type="image/png")

@app.get("/interpolate")
async def interpolate():
    file_path = "./images/interpolate.PNG"
    original_image = Image.open(file_path)
    original_image = original_image.filter(ImageFilter.BLUR)

    filtered_image = BytesIO()
    original_image.save(filtered_image, "PNG")
    filtered_image.seek(0)
    return StreamingResponse(filtered_image, media_type="image/png")
        

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)