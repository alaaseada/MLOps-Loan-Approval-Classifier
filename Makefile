install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_conf_matrix.png)' >> report.md
	
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HUGGING_FACE) --add-to-git-credential

push-hub: 
	huggingface-cli upload alaaseada/MLOps-Loan-Approval-Classifier ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload alaaseada/MLOps-Loan-Approval-Classifier ./Models /Models --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alaaseada/MLOps-Loan-Approval-Classifier ./Results /Metrics --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy