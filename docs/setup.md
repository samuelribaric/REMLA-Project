### 1. Clone the repository
```bash
git clone
```

### 2. Navigate to the project directory
```bash
cd REMLA-Project
```

### 3. Create a virtual environment
```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### On Windows (Powershell)
```bash
venv\Scripts\activate
```

#### On MacOS/Linux
```bash
source venv/bin/activate
```

### 5. Install the dependencies
```bash
pip install -r requirements.txt
```

### 6. Bind to the DVC remote storage
```bash
dvc pull
```