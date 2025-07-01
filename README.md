# Hackathon_diagram_recognition

To use it this repository follow this steps: 

1. first download the weights of the model from [here](https://huggingface.co/caijanfeng/flowmind2digital) and put "model_final_80k.pth" inside flow2digital_poetry/ckpt/ folder.
2. Install poetry (if you haven't)

```
curl -sSL https://install.python-poetry.org | python3 -
```

3. Setup environment
```
cd flow2digital_poetry
poetry env use python3.9
poetry install
poetry shell

```
4. Run app
```
# from flow2digital_poetry folder
python ./src/flow2digital_poetry/API.py 
```

 5. Run inference
```
curl -X POST http://127.0.0.1:5000/process_diagram \
     -H "Content-Type: application/json" \
     -d '{"image_path": "test.png"}'

```
