# Federated HCM diagnosis
A study in multi-center imaging diagnostics, emphasizing on the modality of cardiovascular magnetic resonance and the prediction of hypertrophic cardiomyopathy.

# Usage
Make changes in config.yaml to set your own parameters or model. Use the generate_experiment.py script and note that beside src there will be a new folder named experiments. Each experiment will receive a tag that is a 4 digit number attached to a name you defined. All files from classification will be copied to that folder, so that you have an instance of your experiment which you can then run using the Dockerfile. In code this would be:
```shell
python generate_experiment.py
cd </path/to/your/experiment>
docker build -t experiment .
docker run -v </path/to/datasets>:/root/Datasets -v </path/to/your/experiment>:/experiment experiment
```

# Adapting to another problem
You may adapt this set-up to another diagnostics (or possibly segmentation) problem by tampering with the following files included in the src/classification: config.yaml, models/nets.py, data_loader.py
