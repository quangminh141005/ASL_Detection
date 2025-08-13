# Contributing to ASL Detection Project
---
## 1. Branching Git workflow
1. **Do not work directly from `main`**
2. **Create a new branch** for each person work using your name:
```
git checkout -b model-nguyen-qm
```
3. Make your changes (edit model in `Models/`, add result in `Results`,...) for you work 
4. Commit your change with a message `-m`:
```
git add .
git commit -m"nhan de hieu vao nhe ae:D"
```
5. Push back your branch to GitHub:
```
git push -u origin model-nguyen-qm
```

## 2. Pull Requests (PRs)
1. Open **GitHub**
2. Fill in the PR description:
- Model name
- Brief explanation of the approach and method you are using (CNN, Mediapipe, LSTM,...)
- Pros and cons
- Note for other reviewers (4 other members)
3. Wait for review before merging. If your PR have 2 approvals then your branch can be merge to `main`.

## 3. File Structure
- `/Models/` - model scripts or notebook (`.py` or `.ipynb`, i think we should go for `.ipynb` for learning purpose:D)
- `/Results/` - JSON files with evaluation metrics
- `/data/` - ASL dataset

## 4. Saving Model Results
- After done training your model, save metrics as a **JSON file** in `/Results/`
- Example:
```python
metrics = ASL_history.history
final_metrics = {key: values[-1] for key, values in metrics.items()}

print(final_metrics)

# Save metrics to JSON file
import json
from pathlib import Path

final_metrics["model_name"] = "nguyen quang minh's model"

results_path = Path("/content/ASL_Detection/Results")/"nqm_model.json"
results_path.parent.mkdir(exist_ok=True)

with open(results_path, "w") as f:
  json.dump(final_metrics, f, indent=4)

print(f"Saved results to {results_path}")
```
- Name your `.json` file to your name (e.g.,`nqm_result.json`) to avoid overwrite other member's file

