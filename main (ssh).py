from hand_tracking import hand_classification
from fastapi import FastAPI, Request, Form, Depends, HTTPException, Response
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from io import BytesIO
from datetime import datetime
import numpy as np
import pymysql
import mediapipe as mp
import tensorflow as tf
import cv2
import paramiko
from fastapi.staticfiles import StaticFiles

app = FastAPI()
model = tf.keras.models.load_model("best_model_dnn.h5")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
connected_clients = {}

raspberry_pi_ssh_info = {
    "hostname": "192.168.50.16",
    "port": 22,
    "username": "pi",
    "password": "raspberry"
}

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

class SuccessCounts(BaseModel):
    count1: int
    count2: int
    count3: int
    start_time: float
    name: str

HOST = 'database-1.cje86assyv2g.ap-northeast-2.rds.amazonaws.com'
USER = 'admin'
PASSWORD = 'sessac123'
DATABASE = 'databse12'

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_stream = BytesIO(contents)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)

    predictions = hand_classification(model, image)

    return {"filename": file.filename, "predictions": predictions}


@app.post("/upload-results")
async def upload_results(success_counts: SuccessCounts):
    start_time_datetime = datetime.fromtimestamp(success_counts.start_time).strftime('%Y-%m-%d %H:%M:%S')

    conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO fingerbics (name, success_count1, success_count2, success_count3, datetime) VALUES (%s, %s, %s, %s, %s)",
            (success_counts.name, success_counts.count1, success_counts.count2, success_counts.count3, start_time_datetime)
        )

        print("Data inserted successfully!")
    except pymysql.Error as e:
        print(f"Error inserting data: {repr(e)}")

    # 커밋 및 연결 종료
    conn.commit()
    conn.close()

    response_content = {
        "received_counts": {
            "count1": success_counts.count1,
            "count2": success_counts.count2,
            "count3": success_counts.count3,
            "start_time": start_time_datetime,
            "name": success_counts.name
        }
    }
    print(response_content)

    return JSONResponse(content=response_content)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request = request, name = "im.html")

def execute_raspberry_pi_script(name):
    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(**raspberry_pi_ssh_info)

    # set_name_command = f'export NAME="{name}"'
    # ssh.exec_command(set_name_command)

    # 실행할 스크립트 경로 및 명령어
    script_path = "/home/pi/test/raspberry_pi_client.py"
    command = f"python3 {script_path} {name}"
    stdin, stdout, stderr = ssh.exec_command(command)

    ssh.close()

    # 실행 중 발생한 에러 로그 출력
    error_logs = stderr.read().decode()
    if error_logs:
        print("Error logs:", error_logs)
    # 실행 결과 가져오기
    result = stdout.read().decode()

    return result

@app.post("/step1", response_class=HTMLResponse)
async def read_step1(request: Request, name: str = Form(...)):
    context = {"request": request, "name": name}
    execute_raspberry_pi_script(name)

    return templates.TemplateResponse("step1.html", context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
