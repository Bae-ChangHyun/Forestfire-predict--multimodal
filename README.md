# Forestfire_prediction

1. 프로젝트의 목적 <br>
강원도의 산불 발생 확률을 기상, 지형, 인적 데이터를 이용하여 예측하는 프로젝트.

2. 사전 준비 <br>

아래 구글드라이브에서 두 파일을 다운로드 후, <br>
[raw파일다운로드](https://drive.google.com/file/d/1Kew7kQTDRqo_X_-T-rW06XjGvHvlBEMm/view?usp=drive_link) / 
[asos파일다운로드](https://drive.google.com/file/d/1KfERjVehpwHckMcY6gKZHB8tRyKIegVM/view?usp=drive_link)  <br>
[wiki](https://github.com/Bae-ChangHyun/Forestfire-predict/wiki/Simple-Code-discription)에 기재되어 있는 사전준비를 미리 해놔야 코드가 오류없이 돌아간다.
또한 데이터의 용량이 크기 때문에 용량이 넓은 드라이브에 디렉토리 설정을 하는 것을 추천한다.

3. 실행방법 <br>
train_model.py의 총 실행시간은 대략 2일정도 걸립니다. <br>
단, 멈췄다가 재시작할시 이전 중단지점부터 다시 시작하기 때문에 멈췄다가 다시 실행하여도 상관없습니다. <br>

`train_model.py`의 경우 실행후, 아무것도 입력할 필요없으며, <br>
`test_model.py`의 경우 실행 후, prompt에 안내되는 형식에 따라 input 숫자를 입력하면 됩니다. <br>
(과거시점의 경우 입력되는 날짜, 시간의 산불발생확률맵을 저장, <br> 미래시점(오늘+2일내)의 경우 입력되는 날짜 이후 1시간30,2시간30,3시간30분 뒤의 산불발생확률맵을 저장) <br>
```python
pip install -r requirements.txt
python train_model.py
python test_model.py
```

