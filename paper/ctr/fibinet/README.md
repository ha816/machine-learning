# FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

ACM Conference on Recommender Systems (RecSys '19), 23 May 2019

https://arxiv.org/abs/1905.09433

# Dataset

CTR dataset
- https://www.kaggle.com/c/criteo-display-ad-challenge/data
- http://labs.criteo.com/downloads/2014-kaggle-displayadvertising-challenge-dataset/
  - 위 사이트는 outdate
- 


|                  | criteo-display-ad-challenge |
|:-----------------|:---------------------------:|
| #Trains / #Tests |         1999 / 1999         |

## Data fields

Label - Target variable that indicates if an ad was clicked (1) or not (0).
I1-I13 - A total of 13 columns of integer features (mostly count features).
C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for
anonymization purposes.
