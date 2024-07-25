# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

ADKDD 17, March 13

https://arxiv.org/pdf/1703.04247v1

# Dataset

|                                                          Service                                                          |                                                           Demo                                                            |  
|:-------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------:|  
| <img width="350" alt="image" src="https://media.oss.navercorp.com/user/18854/files/1321d689-86d4-4b05-9c98-44e02e5d90a1"> | <img width="550" alt="image" src="https://media.oss.navercorp.com/user/18854/files/ad194ec9-fd12-4c20-b860-47301f858189"> |

CTR dataset

- https://www.kaggle.com/c/criteo-display-ad-challenge/data
- http://labs.criteo.com/downloads/2014-kaggle-displayadvertising-challenge-dataset/
    - 위 사이트는 outdate

|                  | criteo-display-ad-challenge |
|:-----------------|:---------------------------:|
| #Trains / #Tests |         2000 / 2000         |

## Data fields

Label - Target variable that indicates if an ad was clicked (1) or not (0).
I1-I13 - A total of 13 columns of integer features (mostly count features).
C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for
anonymization purposes.

|       Category       |                     Item                     |
|:--------------------:|:--------------------------------------------:|
| Programming Language |             Java, Python, Scala              |
|      Framework       |         Spring Boot + WebFlux, Spark         |
|       Database       | MySQL, MongoDB, Elasticsearch, Neo4j, Redis  |
|         ETC          | K8s, Argo CD, Helm, Kubeflow, Argo Workflows |