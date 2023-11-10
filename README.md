# Supervised Learning for Hyperparameter Optimization with Grid Search 

모델을 학습할 때 하이퍼 파라미터를 튜닝하는 방법은 여러가지가 있음. 

 - Manual Search
 - Grid Search
 - Random Search
 - Bayesian Optimization
 - Non-Probabilistic
 - Evolutionary Optimization
 - Gradient-based Optimization
 - Early Stopping

이 중 Grid Search를 용이하게 하기 위한 레포지토리임. 사용자는 학습 결과를 Notion에서 확인하면 됨.

## 설명

1. Generate_config.py

학습에 필요한 config.json 파일을 생성하는 함수. 하이퍼 파라미터 pool을 list, 데이터 경로나 로그 경로 등을 초기화하여 for문으로 모든 경우의 수 생성.

2. train.py

Generate_config.py를 실행하여 생성된 모든 경우의 수를 순차적으로 학습을 하는 코드.

```python
from modules import supervised as supervised

JSON_list = natsorted(glob("./json_list/*.json"))

for JSON in JSON_list:
    config = json.load(open(JSON)) # 생성된 하이퍼 파라미터 로드.
    print(JSON)
    Experiment = supervised.Experiment_Model(**config) # 로드한 하이퍼 파라미터로 진행할 실험 초기화.
    Experiment.save_config(JSON) # 로그 디렉토리에 로드한 하이퍼 파라미터 config 저장.
    Experiment.train_model() # 학습 시작
    Experiment.test_model() # 학습 평가
    Experiment.report_model() # 보고서 생성
    Experiment.upload_notion() # 노션 DB에 학습 결과 업로드.
```

3. test.py 

 Threshold를 변경하거나, 테스트 데이터를 변경하여 다시 평가를 해보고 싶을 때 사용하는 코드.

4. test_all.py

로그 디렉토리에 존재하는 모든 학습된 모델들을 다시 평가할 때 사용하는 코드.

5. remove_log.py

학습이 진행되지 않고 로그 디렉토리만 생성되어 자리를 차지하고 있는 학습 로그들을 삭제하는 코드.


## Notion Database

https://github.com/LHyunn/NotionDatabaseAPI 참고.





