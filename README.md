![image](https://github.com/Bae-ChangHyun/Forestfire-predict/assets/48899047/fccf0208-360a-44be-a689-2bc5ff35978b)

# Forestfire_prediction
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBae-ChangHyun%2FForestfire-predict%2Fblob%2Fmain%2FREADME.md&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

개발기간: <br>
`공동`(2023.01 ~ 2023.05) <br>
`개인`(2023.05 ~ 2023.11)

### 1. 프로젝트의 목적 <br>
강원도의 산불 발생 확률을 기상, 지형, 인적 데이터를 이용하여 예측하는 프로젝트.<br>
기상 데이터는 tabular 데이터로, 지형과 인적데이터는 image 데이터로 구성하여, 멀티모달 학습방식을 사용.

### 2. Installation <br>

아래 구글드라이브에서 두 파일을 다운로드 후, <br>
[raw파일다운로드](https://drive.google.com/file/d/1Kew7kQTDRqo_X_-T-rW06XjGvHvlBEMm/view?usp=drive_link) / 
[asos파일다운로드](https://drive.google.com/file/d/1KfERjVehpwHckMcY6gKZHB8tRyKIegVM/view?usp=drive_link)  <br>
[wiki](https://github.com/Bae-ChangHyun/Forestfire-predict/wiki/Simple-Code-discription)에 기재되어 있는 사전준비를 미리 해놔야 코드가 오류없이 돌아간다.
또한 데이터의 용량이 크기 때문에 용량이 넓은 드라이브에 디렉토리 설정을 하는 것을 추천한다.

### 3. Run <br>
`train_model.py`의 총 실행시간은 대략 2일정도 걸립니다. <br>
단, 멈췄다가 재시작할시 이전 중단지점부터 다시 시작하기 때문에 멈췄다가 다시 실행하여도 상관없습니다. <br>

`train_model.py`의 경우 실행후, 아무것도 입력할 필요없으며, <br>
`test_model.py`의 경우 실행 후, prompt에 안내되는 형식에 따라 input 숫자를 입력하면 됩니다. <br>
(과거시점의 경우 입력되는 날짜, 시간의 산불발생확률맵을 저장, <br> 미래시점(오늘+2일내)의 경우 입력되는 날짜 이후 1시간30,2시간30,3시간30분 뒤의 산불발생확률맵을 저장) <br>

```python
pip install -r requirements.txt
python train_model.py
python test_model.py
```
<div align=center><h1>📚 STACKS</h1></div>

<div align=center> 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
  <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/qgis-589632?style=for-the-badge&logo=qgis&logoColor=white">
  <img src="https://img.shields.io/badge/gdal-5CAE58?style=for-the-badge&logo=gdal&logoColor=white">
  <br>
</div>
