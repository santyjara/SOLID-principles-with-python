# 4. Interface Segregation Principle - ISP

## Structure

```
├── 4. Interface Segregation Principle - ISP                   
│   ├── base.py                    # interface and context
│   ├── rnn_model.py               # RNN model strategy
│   ├── sentence_encoder_model.py  # Sentence encoder strategy
│   ├── train.py                   # call model function
│   └── config.yml                 # model parameters
└── ...
```
### Run it locally

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements`

- RNN

`python -m train -m train -t RNN`

- Sentence encoder

`python -m train -m train -t sentence-encoder`